import tensorflow as tf # Você ainda precisará do TF para o TFLite Interpreter
import numpy as np
import cv2
import os

# --- Configurações Globais ---
# Observe que o modelo a ser carregado agora é o .tflite
MODEL_PATH = 'modelo_digitos_mnist.tflite'
# O OUTPUT_FILE e DEFAULT_IMAGE_NAME são relativos à pasta 'api/'
OUTPUT_FILE = 'resultado_predicao.txt'
DEFAULT_IMAGE_NAME = 'meus_digitos.png' # Assumimos que o Flask vai mover o upload para este nome

# --- Parte 1: Preparação e Carregamento do Modelo TFLite ---

def carregar_modelo_tflite():
    """
    Carrega o modelo TFLite. Se não existir, avisa para gerá-lo.
    """
    # Caminho absoluto para o modelo dentro do diretório 'api/'
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

# --- Função Auxiliar para Salvar Resultados ---
def salvar_resultado(texto):
    """Salva o texto fornecido em um arquivo de saída, sobrescrevendo o conteúdo anterior."""
    # O arquivo de resultado também estará dentro da pasta 'api/'
    result_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(result_filepath, 'w') as f:
        f.write(f"{texto}\n")
    print(f"Resultado salvo em: {result_filepath}")


# --- Parte 2: Função para Predição de Múltiplos Dígitos (Padrão) ---

# A função agora recebe o interpreter do TFLite
def prever_multiplos_digitos_em_imagem(interpreter, caminho_imagem):
    """
    Carrega uma imagem com múltiplos dígitos, segmenta, prevê cada um e os junta usando TFLite.
    """
    print(f"\n--- Analisando imagem com múltiplos dígitos: {caminho_imagem} ---")
    # Caminho completo para a imagem que o Flask moveu/renomeou para DEFAULT_IMAGE_NAME
    image_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_IMAGE_NAME)

    try:
        original_img = cv2.imread(image_full_path)
        if original_img is None:
            print(f"Erro: Não foi possível carregar a imagem em {image_full_path}. Verifique o caminho.")
            salvar_resultado(f"IMAGEM: '{image_full_path}' -> Erro: Não foi possível carregar a imagem.")
            return

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_digits_data = []

        # Obter detalhes de entrada do modelo TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # O modelo TFLite espera um tensor de entrada do tipo e forma corretos
        input_shape = input_details[0]['shape'] # Deve ser (1, 28, 28, 1)
        input_dtype = input_details[0]['dtype'] # Geralmente np.float32 ou np.uint8 se for quantizado

        for contour in contours:
            area = cv2.contourArea(contour)
            # Esses valores podem precisar de ajuste fino dependendo da sua imagem e tamanho dos dígitos
            # Mantenha os valores consistentes com o que funcionou antes
            if 10 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                digit_roi = thresh[y:y+h, x:x+w]

                processed_digit = np.zeros((28, 28), dtype=np.uint8)
                if w > h:
                    new_w = 20
                    new_h = int(20 * h / w)
                else:
                    new_h = 20
                    new_w = int(20 * w / h)

                resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                processed_digit[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_digit

                # Normaliza e expande dimensões para o formato do modelo TFLite
                # O tipo de dado deve ser o esperado pelo TFLite interpreter
                processed_digit = processed_digit.astype(input_dtype) / 255.0 if input_dtype == np.float32 else processed_digit
                processed_digit = np.expand_dims(processed_digit, axis=-1)
                processed_digit = np.expand_dims(processed_digit, axis=0)

                detected_digits_data.append({'x': x, 'digit_img': processed_digit})

        detected_digits_data.sort(key=lambda d: d['x'])

        predicted_number_str = ""

        if not detected_digits_data:
            result_text = f"IMAGEM: '{caminho_imagem}' -> Nenhum dígito detectado."
            print(result_text)
            salvar_resultado(result_text)
            return

        # Preve cada dígito usando o TFLite interpreter
        for digit_data in detected_digits_data:
            digit_img_for_prediction = digit_data['digit_img']

            # Define o tensor de entrada e invoca o interpretador
            interpreter.set_tensor(input_details[0]['index'], digit_img_for_prediction)
            interpreter.invoke()

            # Obtém o tensor de saída
            predictions = interpreter.get_tensor(output_details[0]['index'])
            predicted_digit = np.argmax(predictions[0])
            predicted_number_str += str(predicted_digit)

        result_text = f"IMAGEM: '{caminho_imagem}' -> Número Previsto: {predicted_number_str}"
        print(result_text)
        salvar_resultado(result_text)

    except Exception as e:
        print(f"Ocorreu um erro durante a predição de múltiplos dígitos: {e}")
        salvar_resultado(f"IMAGEM: '{caminho_imagem}' -> Erro: {e}")


# --- Execução Principal (Automatizada para Múltiplos Dígitos) ---
if __name__ == "__main__":
    # Carrega o modelo TFLite
    tflite_interpreter = carregar_modelo_tflite()

    if tflite_interpreter:
        print("\n--- Modo de Análise Automático para Múltiplos Dígitos (TFLite) ---")
        print(f"O programa tentará ler a imagem '{DEFAULT_IMAGE_NAME}'.")
        print("Por favor, certifique-se de que a imagem esteja na mesma pasta do script.")
        print("Desenhe os dígitos em um fundo claro (branco) e os números em cor escura (preto).")
        print("Deixe um pequeno espaço entre os dígitos para uma melhor detecção.")

        # Chama a função de predição de múltiplos dígitos com o nome de arquivo padrão
        # (O Flask já terá movido o arquivo carregado para 'meus_digitos.png')
        prever_multiplos_digitos_em_imagem(tflite_interpreter, DEFAULT_IMAGE_NAME)

        print("\nAnálise concluída. Verifique o arquivo 'resultado_predicao.txt' para o resultado.")
    else:
        print("\nNão foi possível carregar o modelo TFLite. Abortando a análise.")
    print("Pressione Enter para sair...")
    input()