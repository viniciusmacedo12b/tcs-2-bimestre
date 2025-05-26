import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# --- Configurações Globais ---
MODEL_PATH = 'modelo_digitos_mnist.keras'   # Nome do arquivo onde o modelo será salvo
OUTPUT_FILE = 'resultado_predicao.txt'      # Nome do arquivo de saída com o resultado
DEFAULT_IMAGE_NAME = 'meus_digitos.png'     # Nome do arquivo de imagem padrão para múltiplos dígitos

# --- Parte 1: Preparação e Treinamento/Carregamento do Modelo ---

def treinar_ou_carregar_modelo_mnist():
    """
    Carrega o modelo se já existir, senão treina um novo modelo e o salva.
    """
    if os.path.exists(MODEL_PATH):
        print(f"--- Carregando modelo existente de: {MODEL_PATH} ---")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo carregado com sucesso!")
        return model
    else:
        print("--- Nenhum modelo existente encontrado. Treinando um novo modelo ---")
        print("Este processo pode levar alguns minutos na primeira execução...")
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # Pré-processamento dos dados para o treinamento da CNN
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        # Construção da Rede Neural Convolucional (CNN)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax') # 10 saídas para dígitos de 0 a 9
        ])

        # Compilação do modelo
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Treinamento do modelo
        model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

        # Avaliação final
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f'\nPrecisão no conjunto de teste: {test_acc:.4f}')

        # Salva o modelo após o treinamento para uso futuro
        model.save(MODEL_PATH)
        print(f"Modelo salvo em: {MODEL_PATH}")
        return model

# --- Função Auxiliar para Salvar Resultados ---

def salvar_resultado(texto):
    """Salva o texto fornecido em um arquivo de saída, sobrescrevendo o conteúdo anterior."""
    # A mudança está aqui: 'w' em vez de 'a'
    with open(OUTPUT_FILE, 'w') as f: # 'w' para sobrescrever o arquivo (write)
        f.write(f"{texto}\n")

# --- Parte 2: Função para Predição de Múltiplos Dígitos (Padrão) ---

def prever_multiplos_digitos_em_imagem(model, caminho_imagem):
    """
    Carrega uma imagem com múltiplos dígitos, segmenta, prevê cada um e os junta.
    """
    try:
        original_img = cv2.imread(caminho_imagem)
        if original_img is None:
            salvar_resultado(f"Erro ao carregar imagem de múltiplos dígitos: {caminho_imagem}")
            return

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Limiarização (Thresholding): converte para preto e branco, invertendo cores
        # cv2.THRESH_OTSU calcula o limiar automaticamente
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Encontrar contornos dos dígitos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_digits_data = []

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filtra contornos por área para remover ruído ou contornos muito grandes
            # Esses valores podem precisar de ajuste fino dependendo da sua imagem e tamanho dos dígitos
            if 10 < area < 20000: # Ajuste estes valores se necessário
                x, y, w, h = cv2.boundingRect(contour)

                # Extrai a Região de Interesse (ROI) do dígito
                digit_roi = thresh[y:y+h, x:x+w]

                # Pré-processa o dígito extraído para o modelo 28x28 com padding e centralização
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

                # Normaliza e expande dimensões para o formato do modelo (1 imagem, 28x28 pixels, 1 canal de cor)
                processed_digit = processed_digit.astype('float32') / 255
                processed_digit = np.expand_dims(processed_digit, axis=-1)
                processed_digit = np.expand_dims(processed_digit, axis=0)

                detected_digits_data.append({'x': x, 'digit_img': processed_digit})

        # Ordena os dígitos da esquerda para a direita com base na coordenada x
        detected_digits_data.sort(key=lambda d: d['x'])

        predicted_number_str = ""

        if not detected_digits_data:
            result_text = f"IMAGEM MÚLTIPLOS DÍGITOS: '{caminho_imagem}' -> Nenhum dígito detectado."
            salvar_resultado(result_text)
            return

        # Preve cada dígito e constrói o número final
        for digit_data in detected_digits_data:
            digit_img_for_prediction = digit_data['digit_img']
            # verbose=0 impede que o Keras imprima cada predição de dígito individual
            predictions = model.predict(digit_img_for_prediction, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            predicted_number_str += str(predicted_digit)

        result_text = f"Número Previsto: {predicted_number_str}"

        salvar_resultado(result_text)

    except Exception as e:
        salvar_resultado(f"Erro na predição de múltiplos dígitos para '{caminho_imagem}': {e}")


# --- Execução Principal (Automatizada para Múltiplos Dígitos) ---

if __name__ == "__main__":
    # Carrega ou treina o modelo de IA
    modelo_treinado = treinar_ou_carregar_modelo_mnist()


    # Chama a função de predição de múltiplos dígitos com o nome de arquivo padrão
    prever_multiplos_digitos_em_imagem(modelo_treinado, DEFAULT_IMAGE_NAME)