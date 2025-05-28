from flask import Flask, request, jsonify
import subprocess
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configurações Globais ---
# Estes caminhos são relativos ao DIRETÓRIO ONDE app.py está (ou seja, 'api/')
# Note que PYTHON_SCRIPT agora aponta para o reconhecedor_digitos_unificado.py
PYTHON_SCRIPT = 'main.py'
RESULT_FILE = 'resultado_predicao.txt'
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# O nome da imagem padrão que o script de reconhecimento vai ler
# Agora, é o arquivo 'meus_digitos.png' que estará DENTRO DA PASTA 'api/'
DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meus_digitos.png')

# Configura o Flask para saber onde salvar os uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Cria a pasta de uploads se não existir ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Funções Auxiliares (mantidas as mesmas do app.py anterior) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# A função salvar_resultado original é do reconhecedor_digitos_unificado.py
# O app.py apenas lê o resultado.


# --- Rota para a API de predição ---
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

        # Copia o arquivo enviado para o nome padrão esperado pelo script de reconhecimento
        try:
            if os.path.exists(DEFAULT_IMAGE_PATH):
                os.remove(DEFAULT_IMAGE_PATH)
            os.rename(filepath, DEFAULT_IMAGE_PATH)
            print(f"'{filename}' movido para '{DEFAULT_IMAGE_PATH}'.")
        except Exception as e:
            print(f"Erro ao mover/renomear arquivo para 'meus_digitos.png': {e}")
            return jsonify({'success': False, 'message': f"Erro interno ao processar a imagem: {e}"}), 500

        try:
            # Caminho para o arquivo de resultado dentro da pasta 'api/'
            result_filepath_for_reading = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_FILE)
            if os.path.exists(result_filepath_for_reading):
                with open(result_filepath_for_reading, 'w') as f:
                    f.write('') # Limpa o arquivo antes de rodar o script

            print(f"Executando script Python: {PYTHON_SCRIPT}")
            # Certifique-se de que o subprocesso chame o script correto
            script_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PYTHON_SCRIPT)
            process = subprocess.run(['python', script_full_path],
                                     capture_output=True, text=True, check=True)

            time.sleep(1) # Aumentar o tempo para 1 segundo pode ser mais seguro na Vercel

            predicted_number = "Erro ao processar resultado."
            if os.path.exists(result_filepath_for_reading):
                with open(result_filepath_for_reading, 'r') as f:
                    result_line = f.read().strip()
                print(f"Resultado lido do arquivo: {result_line}")

                if "Número Previsto: " in result_line:
                    predicted_number = result_line.split("Número Previsto: ")[1]
                elif "Nenhum dígito detectado." in result_line:
                    predicted_number = "Nenhum dígito detectado na imagem."
                elif "Erro:" in result_line:
                    predicted_number = f"Erro no script: {result_line.split('Erro: ')[1]}"
                else:
                    predicted_number = "Formato de resultado inesperado."
            else:
                predicted_number = "Erro: Arquivo de resultado não encontrado."
                print(f"Erro: {RESULT_FILE} não encontrado após a execução do script.")

            return jsonify({'success': True, 'prediction': predicted_number})

        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o script Python: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return jsonify({'success': False, 'message': f"Erro ao executar script: {e.stderr.strip()}"}), 500
        except Exception as e:
            print(f"Erro inesperado no servidor: {e}")
            return jsonify({'success': False, 'message': f"Erro interno no servidor: {e}"}), 500
    else:
        return jsonify({'success': False, 'message': 'Tipo de arquivo não permitido. Por favor, envie PNG, JPG, JPEG ou GIF.'}), 400

# O bloco __name__ == '__main__' é principalmente para execução LOCAL
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    print("Servidor Flask iniciado localmente. Acesse: http://127.0.0.1:5000/")
    # Para testar localmente, você pode querer que o app.py sirva o HTML embutido
    LOCAL_INDEX_HTML_CONTENT = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reconhecedor de Dígitos por IA</title>
        <style>
            body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f0f0f0; margin: 0; flex-direction: column; }
            .container { background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align: center; width: 350px; }
            input[type="file"] { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; width: calc(100% - 22px); box-sizing: border-box; }
            button { padding: 15px 30px; font-size: 1.2em; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s ease; width: 100%; }
            button:hover { background-color: #45a049; }
            button:disabled { background-color: #cccccc; cursor: not-allowed; }
            #loading { margin-top: 20px; font-size: 1.1em; color: #555; display: none; }
            #fileWarning { color: red; margin-top: 10px; font-size: 0.9em; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reconhecedor de Dígitos com IA</h1>
            <p>Selecione uma imagem (PNG, JPG, JPEG, GIF) para análise:</p>
            <input type="file" id="imageUpload" name="image" accept="image/png, image/jpeg, image/jpg, image/gif">
            <div id="fileWarning">Por favor, selecione um arquivo de imagem.</div>
            <button id="predictButton">Analisar Imagem</button>
            <div id="loading">Analisando... Por favor, aguarde.</div>
        </div>

        <script>
            document.getElementById('predictButton').addEventListener('click', async () => {
                const button = document.getElementById('predictButton');
                const loadingDiv = document.getElementById('loading');
                const fileInput = document.getElementById('imageUpload');
                const fileWarning = document.getElementById('fileWarning');

                if (fileInput.files.length === 0) {
                    fileWarning.style.display = 'block';
                    return;
                } else {
                    fileWarning.style.display = 'none';
                }

                button.disabled = true;
                loadingDiv.style.display = 'block';

                const formData = new FormData();
                formData.append('image', fileInput.files[0]);

                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.success) {
                        alert(`Previsão da IA: ${data.prediction}`);
                    } else {
                        alert(`Erro na predição: ${data.message}`);
                    }
                } catch (error) {
                    console.error('Erro ao conectar ao servidor:', error);
                    alert('Ocorreu um erro ao tentar se comunicar com o servidor. Verifique se o servidor está rodando ou se a URL da API está correta.');
                } finally {
                    button.disabled = false;
                    loadingDiv.style.display = 'none';
                    fileInput.value = '';
                }
            });
        </script>
    </body>
    </html>
    """
    @app.route('/')
    def local_index():
        return LOCAL_INDEX_HTML_CONTENT
    app.run(debug=True)