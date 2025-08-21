import os
import sys

# Try to add DLL paths
dll_paths = [
    r'C:\Windows\System32',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin',
    r'C:\Program Files\NVIDIA Corporation\NVSMI',
]

for path in dll_paths:
    if os.path.exists(path):
        os.add_dll_directory(path)

print("Testing TensorFlow import...")
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully!")
    print(f"Version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Test basic functionality
    print("Testing basic operations...")
    hello = tf.constant('Hello, TensorFlow!')
    print(hello.numpy().decode())
    
except Exception as e:
    print(f"❌ TensorFlow failed: {e}")
    print("\nTrying CPU-only mode...")
    
    # Force CPU mode
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        import tensorflow as tf
        print("✅ TensorFlow CPU mode works!")
    except Exception as e2:
        print(f"❌ CPU mode also failed: {e2}")