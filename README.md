# AVSE
This project was developed as a part of the Audio-Visual Speech Enhancement Challenge(AVSEC).

We propose a novel audio-visual speech enhancement framework developed for the COG-MHEAR AVSE Challenge.
It features a dual-stream architecture with a UNet-based udio encoder and a Swin Transformer V2 visual encoder, each processing their modality in parallel.
The extracted features are fused using bi-directional cross-attention, refined via Squeezeformer-based temporal modeling, and decoded using a U-Net–style waveform decoder to generate clean speech.
To address the challenges of speech enhancement in noisy real-world environments, we present its core design principles listed below :

• ChatterboxSwinAVSE is a dual-stream, bi-directional audio-visual speech enhancement framework designed to leverage
noise-robust visual cues, synergistically with audio, to overcome limitations of audio-only methods and achieve robust 
speech clarity in degraded environments.

• The audio encoder employs a UNet-like 1D convolutional 
network to directly process raw noisy waveforms, capturing 
multi-scale temporal dependencies for efficiently extracting 
robust, noise-tolerant features without explicit frequency domain conversion. 63
• A Swin Transformer V2 video encoder is utilized to extract 
rich, multi-scale spatial features from silent video frames 
(e.g., lip movements and facial dynamics), providing crucial 
noise-independent visual cues that aid in speech disambiguation. 
• A bi-directional cross-attention mechanism fuses the audio 
and visual features iteratively, enabling mutual contextualiza
tion and forming a unified representation. This fused embedding is further refined through temporal modeling and passed 
to a U-Net–style decoder for waveform reconstruction. 
Together, these components form a unified, cross-modal architecture that combines the strengths of both audio and visual 
inputs while introducing key innovations—such as raw waveform encoding, spatially rich visual processing, and bidirec- 
tional cross-modal fusion.
