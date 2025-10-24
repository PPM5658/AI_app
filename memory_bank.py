# memory_bank.py
# Updated to use IndexFlatIP for Cosine Similarity
# and removed inaccurate distance reporting.

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import config
from utils import format_time
import threading

class MemoryBank:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Get the embedding dimension from the model
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index for Cosine Similarity (Inner Product)
        # We use IndexFlatIP because we will normalize our vectors.
        index_flat = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(index_flat)
        
        self.video_memories = [] # Stores the raw text
        self.detected_objects = {} # Tracks objects by ID
        self.memory_id_counter = 0
        
        # Add the lock for thread-safety
        self.lock = threading.Lock()
        
        print("MemoryBank initialized.")

    def _generate_memory_text(self, track_id, class_id, confidence, timestamp_sec, yolo_model):
        """
        Generates a text description for a detected object.
        
        --- IMPROVEMENT ---
        Removed 'distance_m'. We do not log data we know is 
        inaccurate (from the FOCAL_LENGTH warning).
        """
        
        obj_id = int(track_id)
        class_name_str = yolo_model.model.names[int(class_id)]
        
        time_str = format_time(timestamp_sec)
        memory_text = ""

        if obj_id not in self.detected_objects:
            self.detected_objects[obj_id] = class_name_str
            
            # --- FIXED MEMORY STRING (NO DISTANCE) ---
            memory_text = (
                f"At {time_str}, a new object '{class_name_str}' (ID: {obj_id}) "
                f"appeared with {confidence:.2f} confidence."
            )
            # -----------------------------------------------
            
        return memory_text, obj_id

    def add_frame_memories(self, results, timestamp_sec, yolo_model):
        """
        Processes the 8-column numpy array [x,y,x,y,id,class,conf,dist]
        from the VideoProcessor.
        """
        tracks = results
        
        with self.lock:
            new_memories_text = []
            new_memory_ids = []

            if not tracks.any():
                return []

            for track in tracks:
                try:
                    # Unpack the columns we need
                    track_id = int(track[4])
                    class_id = int(track[5])
                    confidence = float(track[6])
                    # We no longer read track[7] (distance)
                
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Warning: Skipping malformed track data: {track} (Error: {e})")
                    continue
                
                # Call the helper function (which no longer takes distance)
                memory_text, obj_id = self._generate_memory_text(
                    track_id, class_id, confidence, timestamp_sec, yolo_model
                )
                
                if memory_text: # Only add if it's a new object
                    self.video_memories.append(memory_text)
                    new_memories_text.append(memory_text)
                    new_memory_ids.append(self.memory_id_counter)
                    self.memory_id_counter += 1

            # Batch-embed and add to FAISS
            if new_memories_text:
                embeddings = self.embedding_model.encode(new_memories_text)
                
                # --- CRITICAL FOR IndexFlatIP ---
                # Normalize embeddings to unit length for Cosine Similarity
                faiss.normalize_L2(embeddings)
                
                ids_array = np.array(new_memory_ids).astype('int64')
                self.index.add_with_ids(embeddings, ids_array)

            return new_memories_text
            
    def search_memories(self, query, k=5):
        """
        Searches the memory bank for a query and returns text with similarity scores.
        
        --- IMPROVEMENT ---
        Renamed 'distances' to 'scores' for clarity. IndexFlatIP
        returns similarity scores (higher is better), not distances.
        """
        
        with self.lock:
            if self.index.ntotal == 0:
                return []
                
            query_embedding = self.embedding_model.encode([query])
            
            # --- CRITICAL FOR IndexFlatIP ---
            # Normalize the query embedding as well
            faiss.normalize_L2(query_embedding)
            
            # 'scores' will be an array of similarity scores (higher is better)
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            # Use 'score' as the variable name
            for i, score in zip(indices[0], scores[0]):
                if i != -1 and i < len(self.video_memories):
                    # Return a tuple: (text, similarity_score)
                    results.append((self.video_memories[i], score))
                    
            return results