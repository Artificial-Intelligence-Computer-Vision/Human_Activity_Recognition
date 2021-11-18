from header_imports import *

if __name__ == "__main__":
    
    # Begin analysis
    if len(sys.argv) != 1:

        if sys.argv[1] == "activity_recognition_building":
            activity_recognition_obj = activity_recognition_building(model_type = sys.argv[2])

        if sys.argv[1] == "activity_recognition_training":
            activity_recognition_obj = activity_recognition_training(model_type = sys.argv[2])

        if sys.argv[1] == "activity_recognition_model_only":
            pass
        
        if sys.argv[1] == "activity_recognition_with_computer_vision_activity_recognition":
            pass


