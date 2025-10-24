# memory_bank.py
# ------------------------------------------------------------
# This module provides a semantic memory system for detected video objects.
# It encodes visual events into text, embeds them into vector form using
# a SentenceTransformer model, and stores them in a FAISS index for
# similarity-based retrieval during Q&A.
# ------------------------------------------------------------

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
        
        # Internal storage for memory management
        self.video_memories = []      # List of raw text memories
        self.detected_objects = {}    # Maps object IDs to class names (to avoid duplicates)
        self.memory_id_counter = 0    # Assigns unique numeric IDs to each memory
        
        # Thread lock to prevent simultaneous writes from multiple threads
        self.lock = threading.Lock()
        
        print("MemoryBank initialized.")

    def _generate_memory_text(self, track_id, class_id, confidence, timestamp_sec, yolo_model):
        """
        Convert a YOLO detection result into a natural-language description.

        Args:
            track_id (int): Unique object tracking ID.
            class_id (int): YOLO class ID (e.g., person, chair, etc.).
            confidence (float): Detection confidence.
            timestamp_sec (float): Frame timestamp in seconds.
            yolo_model (YOLO): YOLO model object to fetch class names.

        Returns:
            tuple[str, int]: Generated memory text and corresponding object ID.
        """
        obj_id = int(track_id)
        class_name_str = yolo_model.model.names[int(class_id)]
        time_str = format_time(timestamp_sec)
        memory_text = ""

        # Only create a new memory the first time an object ID appears
        if obj_id not in self.detected_objects:
            self.detected_objects[obj_id] = class_name_str
            memory_text = (
                f"At {time_str}, a new object '{class_name_str}' (ID: {obj_id}) "
                f"appeared with {confidence:.2f} confidence."
            )

        return memory_text, obj_id

    def add_frame_memories(self, results, timestamp_sec, yolo_model):
        """
        Add new object detections from a single frame into the memory bank.

        Args:
            results (np.ndarray): YOLO detection array with shape [N, 8].
            timestamp_sec (float): Time of the current frame in seconds.
            yolo_model (YOLO): YOLO model instance (used for class name lookup).

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

            # Iterate through all detected objects
            for track in tracks:
                try:
                    track_id = int(track[4])
                    class_id = int(track[5])
                    confidence = float(track[6])
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Warning: Skipping malformed track data: {track} (Error: {e})")
                    continue

                # Generate descriptive text for new objects
                memory_text, obj_id = self._generate_memory_text(
                    track_id, class_id, confidence, timestamp_sec, yolo_model
                )

                # Only store if it's a new object
                if memory_text:
                    self.video_memories.append(memory_text)
                    new_memories_text.append(memory_text)
                    new_memory_ids.append(self.memory_id_counter)
                    self.memory_id_counter += 1

            # If there are new memory texts, embed and store them
            if new_memories_text:
                embeddings = self.embedding_model.encode(new_memories_text)

                # Normalize to unit length for cosine similarity (required for IndexFlatIP)
                faiss.normalize_L2(embeddings)

                # Map new IDs to embeddings and add to FAISS index
                ids_array = np.array(new_memory_ids).astype('int64')
                self.index.add_with_ids(embeddings, ids_array)

            return new_memories_text

    def search_memories(self, query, k=5):
        """
        Search the memory bank for stored events that semantically match a query.

        Args:
            query (str): User's natural-language question.
            k (int): Number of top results to retrieve.

        Returns:
            list[tuple[str, float]]: Pairs of (memory_text, similarity_score).
        """
        with self.lock:
            # No stored memories
            if self.index.ntotal == 0:
                return []

            # Encode and normalize the query for cosine similarity
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Perform semantic search
            scores, indices = self.index.search(query_embedding, k)

            results = []
            for i, score in zip(indices[0], scores[0]):
                # Filter out invalid indices
                if i != -1 and i < len(self.video_memories):
                    results.append((self.video_memories[i], score))

            return results
