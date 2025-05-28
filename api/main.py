# REMOVA cv2
# import cv2 # <--- REMOVER ISSO!

import tensorflow as tf # ou tflite-runtime
import numpy as np
import os
from PIL import Image # Importar Pillow

# --- Configurações Globais ---
MODEL_PATH = 'modelo_digitos_mnist.tflite'
OUTPUT_FILE = 'resultado_predicao.txt'
DEFAULT_IMAGE_NAME = 'meus_digitos.png'

# --- Carregamento do Modelo TFLite (mesmo código anterior) ---
def carregar_modelo_tflite():
    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_full_path):
        print(f"Erro: Modelo TFLite '{model_full_path}' não encontrado.")
        print("Por favor, execute o script 'convert_model.py' em seu ambiente local para gerar este arquivo.")
        return None
    print(f"--- Carregando modelo TFLite de: {model_full_path} ---")
    interpreter = tf.lite.Interpreter(model_path=model_full_path)
    interpreter.allocate_tensors()
    print("Modelo TFLite carregado com sucesso!")
    return interpreter

# --- Função Auxiliar para Salvar Resultados (mesmo código anterior) ---
def salvar_resultado(texto):
    result_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(result_filepath, 'w') as f:
        f.write(f"{texto}\n")
    print(f"Resultado salvo em: {result_filepath}")

# --- NOVO: Função para encontrar bounding boxes usando NumPy (mais robusta) ---
from scipy.ndimage import label, find_objects # <- Adicionar importação

def find_bounding_boxes_numpy(binary_image):
    """
    Encontra bounding boxes de componentes conectados em uma imagem binária usando SciPy.
    Retorna uma lista de tuplas (x, y, w, h).
    """
    # SciPy label espera imagem onde o objeto é True (ou 1) e o fundo é False (ou 0)
    # Certifique-se que sua `binary_image` tenha os dígitos como 1 e o fundo como 0.
    labeled_array, num_features = label(binary_image)
    objects = find_objects(labeled_array)

    bounding_boxes = []
    for obj_slice in objects:
        y_min, y_max = obj_slice[0].start, obj_slice[0].stop
        x_min, x_max = obj_slice[1].start, obj_slice[1].stop

        w = x_max - x_min
        h = y_max - y_min
        area = w * h

        # Ajuste os limites de área conforme a necessidade para filtrar ruído
        if 10 < area < 20000: # Os mesmos limites de área que você usava
            bounding_boxes.append((x_min, y_min, w, h, area))
    return bounding_boxes


# --- Parte 2: Função para Predição de Múltiplos Dígitos (MODIFICADA) ---

def prever_multiplos_digitos_em_imagem(interpreter, caminho_imagem):
    print(f"\n--- Analisando imagem com múltiplos dígitos: {caminho_imagem} ---")
    image_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_IMAGE_NAME)

    try:
        pil_img = Image.open(image_full_path).convert('L') # Convert to grayscale
        original_img_np = np.array(pil_img) # Convert PIL image to NumPy array

        # Limiarização: Fundo branco, Dígitos pretos (ou tons escuros)
        # Se sua imagem tem fundo claro e dígitos escuros:
        thresh = (original_img_np < 128).astype(np.uint8) * 255 # Pixels escuros (abaixo de 128) viram 255 (branco)
                                                              # Pixels claros (acima de 128) viram 0 (preto)
                                                              # Isso INVERTE a imagem para dígitos brancos em fundo preto
        # Para `find_bounding_boxes_numpy`, precisamos de 1s para o objeto e 0s para o fundo
        binary_for_components = (thresh > 0).astype(int) # Dígitos (255) viram 1, Fundo (0) vira 0

        # Encontrar contornos/bounding boxes usando a função SciPy
        bounding_boxes_data = find_bounding_boxes_numpy(binary_for_components)

        detected_digits_data = []

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']

        for x, y, w, h, area in bounding_boxes_data:
            # Recorta a região de interesse (ROI) da imagem limiarizada original (não a binária)
            # Ou, se `thresh` já estiver com o que você quer, use-o diretamente.
            # Se `thresh` já tem dígitos brancos em fundo preto, use-o.
            digit_roi = thresh[y:y+h, x:x+w]

            # Redimensionamento para 20x20 mantendo a proporção (mesma lógica do OpenCV)
            processed_digit_pil = Image.fromarray(digit_roi)
            if w > h:
                new_w = 20
                new_h = int(20 * h / w)
            else:
                new_h = 20
                new_w = int(20 * w / h)
            resized_digit_pil = processed_digit_pil.resize((new_w, new_h), Image.LANCZOS) # Melhor interpolação

            # Criar tela 28x28 e colar o dígito redimensionado
            processed_digit_np = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            processed_digit_np[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = np.array(resized_digit_pil)

            # Normaliza e expande dimensões para o formato do modelo TFLite
            processed_digit_np = processed_digit_np.astype(input_dtype) / 255.0 if input_dtype == np.float32 else processed_digit_np
            processed_digit_np = np.expand_dims(processed_digit_np, axis=-1)
            processed_digit_np = np.expand_dims(processed_digit_np, axis=0) # Adiciona dimensão de batch

            detected_digits_data.append({'x': x, 'digit_img': processed_digit_np})

        detected_digits_data.sort(key=lambda d: d['x'])

        predicted_number_str = ""

        if not detected_digits_data:
            result_text = f"IMAGEM: '{caminho_imagem}' -> Nenhum dígito detectado."
            print(result_text)
            salvar_resultado(result_text)
            return

        for digit_data in detected_digits_data:
            digit_img_for_prediction = digit_data['digit_img']
            interpreter.set_tensor(input_details[0]['index'], digit_img_for_prediction)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_digit = np.argmax(predictions[0])
            predicted_number_str += str(predicted_digit)

        result_text = f"IMAGEM: '{caminho_imagem}' -> Número Previsto: {predicted_number_str}"
        print(result_text)
        salvar_resultado(result_text)

    except Exception as e:
        print(f"Ocorreu um erro durante a predição de múltiplos dígitos: {e}")
        salvar_resultado(f"IMAGEM: '{caminho_imagem}' -> Erro: {e}")

# --- Execução Principal (mesmo código anterior) ---
if __name__ == "__main__":
    tflite_interpreter = carregar_modelo_tflite()
    if tflite_interpreter:
        print("\n--- Modo de Análise Automático para Múltiplos Dígitos (TFLite) ---")
        print(f"O programa tentará ler a imagem '{DEFAULT_IMAGE_NAME}'.")
        print("Por favor, certifique-se de que a imagem esteja na mesma pasta do script.")
        print("Desenhe os dígitos em um fundo claro (branco) e os números em cor escura (preto).")
        print("Deixe um pequeno espaço entre os dígitos para uma melhor detecção.")
        prever_multiplos_digitos_em_imagem(tflite_interpreter, DEFAULT_IMAGE_NAME)
        print("\nAnálise concluída. Verifique o arquivo 'resultado_predicao.txt' para o resultado.")
    else:
        print("\nNão foi possível carregar o modelo TFLite. Abortando a análise.")
    print("Pressione Enter para sair...")
    input()