import tensorflow as tf
import os

MODEL_PATH_KERAS = 'modelo_digitos_mnist.keras'
MODEL_PATH_TFLITE = 'modelo_digitos_mnist.tflite'

def convert_keras_to_tflite():
    if not os.path.exists(MODEL_PATH_KERAS):
        print(f"Erro: Modelo Keras '{MODEL_PATH_KERAS}' não encontrado.")
        print("Certifique-se de que o script de treinamento (reconhecedor_digitos_unificado.py na primeira execução) já foi rodado e salvou o modelo Keras.")
        return

    print(f"Carregando modelo Keras de: {MODEL_PATH_KERAS}")
    model = tf.keras.models.load_model(MODEL_PATH_KERAS)
    print("Modelo Keras carregado com sucesso.")

    # Cria o TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- Otimizações ---
    # Ativa a otimização padrão (quantização)
    # Isso tentará quantizar o modelo para reduzir o tamanho.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Opcional: Se você quiser garantir quantização INT8, pode ser mais agressivo.
    # Para isso, você precisaria de um 'representative_dataset' para calibração.
    # Exemplo:
    # def representative_dataset_generator():
    #     # Carregue alguns dados de treinamento aqui para calibração
    #     # Por exemplo, as primeiras 100 amostras do MNIST
    #     (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    #     train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    #     for i in range(100):
    #         yield [train_images[i:i+1]]
    # converter.representative_dataset = representative_dataset_generator
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8  # Input tensor type for inference
    # converter.inference_output_type = tf.uint8 # Output tensor type for inference


    print("Convertendo modelo para TFLite com otimização padrão...")
    tflite_model = converter.convert()

    # Salva o modelo TFLite
    with open(MODEL_PATH_TFLITE, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo TFLite salvo em: {MODEL_PATH_TFLITE}")
    print(f"Tamanho do modelo Keras: {os.path.getsize(MODEL_PATH_KERAS) / (1024*1024):.2f} MB")
    print(f"Tamanho do modelo TFLite: {os.path.getsize(MODEL_PATH_TFLITE) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_keras_to_tflite()