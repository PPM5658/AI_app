# memory_bank.py
# --------------------------------------------------------------------
# Maintains a semantic + visual memory of detected video objects.
# Extended to store metadata for visual synchronization (for highlight capture).
# --------------------------------------------------------------------

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config
from utils import format_time
import threading


class MemoryBank:
    """
    MemoryBank maintains an embedding-based memory of detected video events.
    It supports:
      - Adding new object memories per frame
      - Encoding textual descriptions into embeddings
      - Searching for relevant past memories given a natural-language query
      - Linking text memories to visual metadata for later highlighting
    """

    def __init__(self):
        """Initialize the embedding model, FAISS index, and thread-safe structures."""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Get the embedding vector size (e.g., 384 or 768)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index for cosine similarity (via normalized inner product)
        index_flat = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(index_flat)

        # Internal storage
        self.video_memories = []      # List[str] of textual memories
        self.memory_metadata = []     # List[dict] of metadata for each memory
        self.detected_objects = {}    # Maps track_id → class name
        self.memory_id_counter = 0    # Assigns unique numeric IDs to each memory

        # Thread lock for concurrency safety
        self.lock = threading.Lock()

        print("MemoryBank initialized successfully.")

    # ----------------------------------------------------------------
    # Private helper: generate memory text
    # ----------------------------------------------------------------
    def _generate_memory_text(self, track_id, class_id, confidence, timestamp_sec, yolo_model):
        """
        Convert a YOLO detection into natural-language description.

        Args:
            track_id (int): Unique object tracking ID.
            class_id (int): YOLO class ID (e.g., person, chair, etc.).
            confidence (float): Detection confidence.
            timestamp_sec (float): Frame timestamp in seconds.
            yolo_model (YOLO): YOLO model object.

        Returns:
            tuple[str, int]: (memory_text, object_id)
        """
        obj_id = int(track_id)
        class_name_str = yolo_model.model.names[int(class_id)]
        time_str = format_time(timestamp_sec)
        memory_text = ""

        # Only add a new memory when an object appears first time
        if obj_id not in self.detected_objects:
            self.detected_objects[obj_id] = class_name_str
            memory_text = (
                f"At {time_str}, a new object '{class_name_str}' (ID: {obj_id}) "
                f"appeared with {confidence:.2f} confidence."
            )

        return memory_text, obj_id, class_name_str

    # ----------------------------------------------------------------
    # Add frame detections to memory
    # ----------------------------------------------------------------
    def add_frame_memories(self, results, timestamp_sec, yolo_model):
        """
        Add new detections from one frame into the memory bank.

        Args:
            results (np.ndarray): YOLO detection array [N, 8].
            timestamp_sec (float): Time of current frame.
            yolo_model (YOLO): YOLO model instance.

        Returns:
            list[str]: Newly added memory text strings.
        """
        tracks = results

        with self.lock:
            new_memories_text = []
            new_memory_ids = []

            # Skip empty or invalid detection results
            if not tracks.any():
                return []

            for track in tracks:
                try:
                    track_id = int(track[4])
                    class_id = int(track[5])
                    confidence = float(track[6])
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Warning: Skipping malformed track data: {track} (Error: {e})")
                    continue

                # Generate descriptive text
                memory_text, obj_id, class_name_str = self._generate_memory_text(
                    track_id, class_id, confidence, timestamp_sec, yolo_model
                )

                # Only store first-time detections
                if memory_text:
                    self.video_memories.append(memory_text)
                    new_memories_text.append(memory_text)
                    new_memory_ids.append(self.memory_id_counter)

                    # --- 🔹 Add structured metadata for this memory ---
                    self.memory_metadata.append({
                        "id": self.memory_id_counter,
                        "object_id": obj_id,
                        "class_name": class_name_str,
                        "confidence": confidence,
                        "timestamp_sec": timestamp_sec
                    })

                    self.memory_id_counter += 1

            # Embed & store new memories in FAISS
            if new_memories_text:
                embeddings = self.embedding_model.encode(new_memories_text)
                faiss.normalize_L2(embeddings)
                ids_array = np.array(new_memory_ids).astype("int64")
                self.index.add_with_ids(embeddings, ids_array)

            return new_memories_text

    # ----------------------------------------------------------------
    # Search for relevant past memories
    # ----------------------------------------------------------------
    def search_memories(self, query, k=5):
        """
        Search the memory bank for stored events that semantically match a query.

        Args:
            query (str): User's natural-language question.
            k (int): Number of top results to retrieve.

        Returns:
            list[tuple[str, float]]: (memory_text, similarity_score)
        """
        with self.lock:
            if self.index.ntotal == 0:
                return []

            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            scores, indices = self.index.search(query_embedding, k)
            results = []
            for i, score in zip(indices[0], scores[0]):
                if i != -1 and i < len(self.video_memories):
                    results.append((self.video_memories[i], score))
            return results

    # ----------------------------------------------------------------
    # Find metadata for a given object label
    # ----------------------------------------------------------------
    def find_object_metadata(self, label):
        """
        Retrieve metadata for the most recent occurrence of an object label.
        Returns None if not found.
        """
        label = label.lower()
        with self.lock:
            for meta in reversed(self.memory_metadata):
                if meta["class_name"].lower() == label:
                    return meta
        return None

    # ----------------------------------------------------------------
    # Debug / utility
    # ----------------------------------------------------------------
    def summary(self):
        """Print summary of stored memories and metadata."""
        with self.lock:
            print("\n=== MemoryBank Summary ===")
            print(f"Total Memories: {len(self.video_memories)}")
            print(f"Unique Objects: {len(self.detected_objects)}")
            if self.memory_metadata:
                last = self.memory_metadata[-1]
                print(
                    f"Last Added → {last['class_name']} "
                    f"(ID {last['object_id']}) at {format_time(last['timestamp_sec'])}"
                )
            print("===========================\n")
