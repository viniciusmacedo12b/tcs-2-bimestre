from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configurações Globais ---
PYTHON_SCRIPT = 'main.py'
RESULT_FILE = 'resultado_predicao.txt'
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meus_digitos.png')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Nenhum arquivo de imagem foi enviado.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'Nenhum arquivo selecionado.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Arquivo '{filename}' salvo em '{filepath}'.")

        try:
            # Garante que o diretório de destino exista
            dest_dir = os.path.dirname(DEFAULT_IMAGE_PATH)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            if os.path.exists(DEFAULT_IMAGE_PATH):
                os.remove(DEFAULT_IMAGE_PATH)
            os.rename(filepath, DEFAULT_IMAGE_PATH)
            print(f"'{filename}' movido para '{DEFAULT_IMAGE_PATH}'.")
        except Exception as e:
            print(f"Erro ao mover/renomear arquivo para 'meus_digitos.png': {e}")
            return jsonify({'success': False, 'message': f"Erro interno ao processar a imagem: {e}"}), 500

        # Caminho completo para o script Python e o arquivo de resultado
        script_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PYTHON_SCRIPT)
        result_filepath_for_reading = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_FILE)

        try:
            # Limpa o arquivo de resultado antes de rodar o script para evitar leitura de dados antigos
            if os.path.exists(result_filepath_for_reading):
                with open(result_filepath_for_reading, 'w') as f:
                    f.write('') # Escreve string vazia para limpar

            print(f"Executando script Python: {PYTHON_SCRIPT}")
            # Use subprocess.run e capture a saída diretamente
            # check=True fará com que um CalledProcessError seja levantado se o script retornar erro
            process_result = subprocess.run(['python', script_full_path],
                                             capture_output=True, text=True, check=True)

            print(f"Stdout do script main.py:\n{process_result.stdout}")
            if process_result.stderr:
                print(f"Stderr do script main.py:\n{process_result.stderr}")

            # Tenta ler o arquivo de resultado com um pequeno retry/timeout
            max_retries = 5
            retry_delay = 0.5 # segundos
            predicted_number = "Erro ao processar resultado."

            for i in range(max_retries):
                if os.path.exists(result_filepath_for_reading) and os.path.getsize(result_filepath_for_reading) > 0:
                    try:
                        with open(result_filepath_for_reading, 'r', encoding='utf-8') as f: # Especifique encoding
                            result_line = f.read().strip()
                        print(f"Resultado lido do arquivo: {result_line}")

                        if "Número Previsto: " in result_line:
                            predicted_number = result_line.split("Número Previsto: ")[1].strip()
                        elif "Nenhum dígito detectado." in result_line:
                            predicted_number = "Nenhum dígito detectado na imagem."
                        elif "Erro:" in result_line:
                            predicted_number = f"Erro no script: {result_line.split('Erro: ')[1].strip()}"
                        else:
                            predicted_number = "Formato de resultado inesperado do script Python."
                        break # Sai do loop se a leitura for bem-sucedida
                    except Exception as file_read_error:
                        print(f"Erro ao ler o arquivo de resultado na tentativa {i+1}: {file_read_error}")
                        time.sleep(retry_delay)
                else:
                    print(f"Arquivo de resultado vazio ou não encontrado na tentativa {i+1}. Aguardando...")
                    time.sleep(retry_delay)
            else: # Executado se o loop não for quebrado (ou seja, todas as retries falharam)
                predicted_number = "Erro: O arquivo de resultado não pôde ser lido ou estava vazio após várias tentativas."
                print(f"Erro: {RESULT_FILE} não pôde ser lido ou estava vazio após {max_retries} tentativas.")
            print(f"Valor final de predicted_number: {predicted_number}")
            return jsonify({'success': True, 'prediction': predicted_number})
            print(f"JSON de resposta que será enviado: {response_data}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o script Python: {e}")
            print(f"Stdout do erro: {e.stdout}")
            print(f"Stderr do erro: {e.stderr}")
            # Se o script falhou com um erro, o output do erro já está no Stderr
            return jsonify({'success': False, 'message': f"Erro ao executar script: {e.stderr.strip()}"}), 500
        except Exception as e:
            print(f"Erro inesperado no servidor: {e}")
            return jsonify({'success': False, 'message': f"Erro interno no servidor: {e}"}), 500
    else:
        return jsonify({'success': False, 'message': 'Tipo de arquivo não permitido. Por favor, envie PNG, JPG, JPEG ou GIF.'}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    print("Servidor Flask iniciado localmente. Acesse: http://127.0.0.1:5000/")
    # Removendo o LOCAL_INDEX_HTML_CONTENT para evitar confusão com o Live Server
    app.run(debug=True, port=5001) # Mude para 5001