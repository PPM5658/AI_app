# 🧠 InsightVision AI  
### *Real-Time Visual Q&A Assistant Powered by YOLOv8, FAISS & Gemini LLM*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Gemini API](https://img.shields.io/badge/Google-Gemini_Pro-yellow.svg)](https://deepmind.google/technologies/gemini/)
[![FAISS](https://img.shields.io/badge/Vector_Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

---

## 📘 Overview
**InsightVision AI** is a real-time **multimodal intelligence system** that merges **computer vision**, **semantic memory**, and **language reasoning** to understand and explain visual scenes.

It:
- Detects and tracks objects using **YOLOv8**
- Estimates object distance via **camera calibration**
- Stores semantic “memories” with **SentenceTransformers + FAISS**
- Answers natural-language questions using **Gemini LLM**
- Responds with **highlighted frames and textual explanations**

> 💬 *Example:*  
> **User:** “Where is the chair?”  
> **AI:** *Highlights the chair and replies:* “The chair is near the left wall, 2.3 m away.”

---

## 🚀 Core Features

| Feature | Description |
|----------|--------------|
| 🎥 **Real-Time Detection** | YOLOv8 identifies and tracks multiple objects in video or webcam feed |
| 📏 **Distance Estimation** | Uses focal-length calibration for approximate real-world distances |
| 🧠 **Semantic Memory Bank** | Stores contextual object information with FAISS + SentenceTransformers |
| 💬 **Natural-Language Q&A** | Gemini LLM interprets visual context to generate accurate answers |
| 🔍 **Smart Frame Retrieval** | Recovers relevant frames to answer late queries |
| 🪟 **Interactive Tkinter GUI** | Clean interface for controlling video, progress, and queries |
| ☁️ **Auto-Save to Drive** | Saves annotated frames to local or Google Drive |
| ⚙️ **Concurrent Threads** | Video and Q&A run simultaneously for smooth real-time interaction |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["🎥 Video Input / Webcam"] --> B["🔍 YOLOv8 Object Detection"]
    B --> C["📏 Distance Estimation"]
    B --> D["🧠 MemoryBank (FAISS + SentenceTransformer)"]
    D --> E["💬 Gemini Q&A System"]
    E --> F["🖼️ Highlighted Frames + Text Answers"]
    F --> G["🪟 Tkinter GUI Interface"]
