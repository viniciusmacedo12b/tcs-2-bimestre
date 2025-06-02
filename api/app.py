import os
import subprocess
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- Configuração do Flask ---

# Determina a pasta raiz do projeto.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Caminho para a pasta onde as imagens enviadas serão salvas temporariamente.
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define o caminho para o modelo TFLite e o script principal de predição.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_digitos_mnist.tflite')
MAIN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'main.py')
RESULT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'resultado_predicao.txt')
TEMP_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'meus_digitos.png')

# Inicializa o aplicativo Flask.
app = Flask(__name__, static_folder=project_root, static_url_path='/')

# Configuração do CORS (Cross-Origin Resource Sharing).
# Use "*" para teste local, mas seja específico para produção no Render.
CORS(app, resources={r"/api/*": {"origins": [
    # "http://127.0.0.1:5500",  # Remover para deploy se não for mais usar o Live Server
    # "null",                   # Remover para deploy se não for mais abrir arquivos locais diretamente
    # "http://127.0.0.1:5000",  # Remover para deploy se não for mais testar localmente assim
    # "*"                       # Remover para deploy (ou comente), pois é muito permissivo
    "https://tcs-2-bimestre.onrender.com" # <<-- MUDE ESTE PARA A URL REAL DO SEU SERVIÇO NO RENDER!
]}})

# --- Rotas da Aplicação ---

# Rota para servir o arquivo HTML principal (index.html)
@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory(project_root, 'index.html')

# Rota para a API de predição
@app.route('/api/predict', methods=['POST'])
def predict():
    # --- INÍCIO DA FUNÇÃO PREDICT ---
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Nenhuma imagem foi enviada.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Nenhum arquivo selecionado.'}), 400

    if file:
        original_filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(filepath)
        print(f"Arquivo '{original_filename}' salvo em '{filepath}'")

        try:
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
            os.rename(filepath, TEMP_IMAGE_PATH)
            print(f"'{original_filename}' movido para '{TEMP_IMAGE_PATH}'")
        except Exception as e:
            print(f"Erro ao mover arquivo: {e}")
            return jsonify({'success': False, 'message': f"Erro ao preparar imagem para predição: {e}"}), 500

        max_retries = 5
        retry_delay = 1
        predicted_number = None

        for i in range(max_retries):
            print(f"Executando script Python: {MAIN_SCRIPT_PATH} (Tentativa {i+1})")
            try:
                result = subprocess.run(
                    ['python', MAIN_SCRIPT_PATH],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("Stdout do script main.py:")
                print(result.stdout)
                if result.stderr:
                    print("Stderr do script main.py:")
                    print(result.stderr)

                file_read_attempts = 0
                while file_read_attempts < 10:
                    if os.path.exists(RESULT_FILE_PATH) and os.path.getsize(RESULT_FILE_PATH) > 0:
                        try:
                            with open(RESULT_FILE_PATH, 'r', encoding='utf-8') as f:
                                line = f.readline().strip()
                                if "Número Previsto:" in line:
                                    predicted_number = line.split("Número Previsto:")[1].strip()
                                    print(f"Resultado lido do arquivo: {line}")
                                    break
                                else:
                                    print(f"Formato inesperado no arquivo '{RESULT_FILE_PATH}': {line}")
                        except UnicodeDecodeError as e:
                            print(f"Erro de decodificação ao ler '{RESULT_FILE_PATH}': {e}. Conteúdo: {line}")
                        except Exception as e:
                            print(f"Erro ao ler '{RESULT_FILE_PATH}': {e}")
                    else:
                        print(f"Arquivo '{RESULT_FILE_PATH}' não encontrado ou vazio. Tentativa {file_read_attempts + 1} de 10.")
                    time.sleep(0.5)
                    file_read_attempts += 1

                if predicted_number:
                    break
                else:
                    print(f"Não foi possível ler a predição do arquivo '{RESULT_FILE_PATH}' após múltiplas tentativas.")

            except subprocess.CalledProcessError as e:
                print(f"Erro ao executar main.py: {e}")
                print(f"Stdout do main.py: {e.stdout}")
                print(f"Stderr do main.py: {e.stderr}")
                predicted_number = f"Erro ao executar script: {e.stderr or e.stdout}"
                break
            except Exception as e:
                print(f"Erro inesperado durante a execução do main.py: {e}")
                predicted_number = f"Erro inesperado: {str(e)}"
                break

            if i < max_retries - 1:
                print(f"Tentando novamente em {retry_delay} segundos...")
                time.sleep(retry_delay)

        try:
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
            if os.path.exists(RESULT_FILE_PATH):
                os.remove(RESULT_FILE_PATH)
        except Exception as e:
            print(f"Erro durante a limpeza de arquivos temporários: {e}")

        if predicted_number:
            response_data = {'success': True, 'prediction': predicted_number}
            print(f"JSON de resposta que será enviado: {response_data}")
            return jsonify(response_data)
        else:
            return jsonify({'success': False, 'message': predicted_number or 'Não foi possível obter uma predição válida.'}), 500
    # --- FIM DA FUNÇÃO PREDICT ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)