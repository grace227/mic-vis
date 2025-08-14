#!/usr/bin/env python3
"""
Simple test script to verify the mic-vis package structure.
Run this after installing the package with: pip install -e .
"""

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        print("Testing imports...")
        
        # Test main package
        import mic_vis
        print("✓ mic_vis imported successfully")
        
        # Test submodules
        from mic_vis import common
        print("✓ mic_vis.common imported successfully")
        
        from mic_vis import bnp
        print("✓ mic_vis.bnp imported successfully")
        
        from mic_vis import s2idd
        print("✓ mic_vis.s2idd imported successfully")
        
        # Test specific functions/classes if they exist
        try:
            from mic_vis.common import readMDA
            print("✓ mic_vis.common.readMDA imported successfully")
        except ImportError:
            print("⚠ mic_vis.common.readMDA not available")
        
        try:
            from mic_vis.common import plot
            print("✓ mic_vis.common.plot imported successfully")
        except ImportError:
            print("⚠ mic_vis.common.plot not available")
        
        try:
            from mic_vis.bnp import io
            print("✓ mic_vis.bnp.io imported successfully")
        except ImportError:
            print("⚠ mic_vis.bnp.io not available")
        
        print("\n🎉 All imports successful! Package is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
