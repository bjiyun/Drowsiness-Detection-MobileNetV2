import tensorflow as tf

for path in [
    "models/eye_model.tflite",
    "models/mouth_model.tflite"
]:
    print("\n", path)

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    print(interpreter.get_input_details())
    print(interpreter.get_output_details())