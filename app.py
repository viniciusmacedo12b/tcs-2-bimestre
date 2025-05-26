from flask import Flask, render_template, request, jsonify
import subprocess
import os
import time
from werkzeug.utils import secure_filename # Importar para nomes de arquivo seguros

app = Flask(__name__)

# --- Configurações Globais ---
PYTHON_SCRIPT = 'main.py'
RESULT_FILE = 'resultado_predicao.txt'
# O arquivo de imagem padrão agora é o TEMPORÁRIO que será salvo
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads') # Cria uma pasta 'uploads' no diretório atual
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Configura o Flask para saber onde salvar os uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Cria a pasta de uploads se não existir ---
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Funções Auxiliares ---
def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def salvar_resultado(texto):
    """Salva o texto fornecido em um arquivo de saída, sobrescrevendo o conteúdo anterior."""
    with open(RESULT_FILE, 'w') as f:
        f.write(f"{texto}\n")
    print(f"Resultado salvo em: {RESULT_FILE}")

# --- Rotas ---
@app.route('/')
def index():
    """Rota para servir a página HTML principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para receber a imagem, executar o script Python e retornar o resultado.
    """
    # 1. Verifica se um arquivo foi enviado
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Nenhum arquivo de imagem foi enviado.'}), 400

    file = request.files['image']

    # 2. Verifica se o nome do arquivo está vazio
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Nenhum arquivo selecionado.'}), 400

    # 3. Processa o arquivo se ele for permitido
    if file and allowed_file(file.filename):
        # Garante um nome de arquivo seguro para evitar problemas de caminho
        filename = secure_filename(file.filename)
        # Caminho completo para salvar o arquivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Arquivo '{filename}' salvo em '{filepath}'.")

        # Agora, modificamos o script Python para que ele leia este novo arquivo
        # Em vez de modificar o script Python em tempo real,
        # vamos passá-lo como um argumento ou criar um link simbólico,
        # ou, mais simples, copiar o arquivo enviado para o nome padrão 'meus_digitos.png'.
        # Isso evita modificar o script 'reconhecedor_digitos_unificado.py'.

        # Copia o arquivo enviado para o nome padrão esperado pelo script de reconhecimento
        target_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meus_digitos.png')
        try:
            # Remove o arquivo antigo se existir
            if os.path.exists(target_image_path):
                os.remove(target_image_path)
            # Renomeia/move o arquivo recém-salvo para 'meus_digitos.png'
            os.rename(filepath, target_image_path)
            print(f"'{filename}' movido para '{target_image_path}'.")
        except Exception as e:
            print(f"Erro ao mover/renomear arquivo para 'meus_digitos.png': {e}")
            return jsonify({'success': False, 'message': f"Erro interno ao processar a imagem: {e}"}), 500

        try:
            # Limpa o arquivo de resultado antes de rodar o script
            if os.path.exists(RESULT_FILE):
                with open(RESULT_FILE, 'w') as f:
                    f.write('') # Escreve um arquivo vazio para limpar

            print(f"Executando script Python: {PYTHON_SCRIPT}")
            process = subprocess.run(['python', PYTHON_SCRIPT], capture_output=True, text=True, check=True)

            # Pequena pausa para garantir que o sistema de arquivos tenha tempo de salvar
            time.sleep(0.5)

            # Lê o resultado do arquivo de texto gerado pelo script Python
            if os.path.exists(RESULT_FILE):
                with open(RESULT_FILE, 'r') as f:
                    result_line = f.read().strip()
                print(f"Resultado lido do arquivo: {result_line}")
                if "Número Previsto: " in result_line:
                    predicted_number = result_line.split("Número Previsto: ")[1]
                else:
                    predicted_number = "Não detectado ou formato inesperado."
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

if __name__ == '__main__':
    # Cria a pasta 'templates' se não existir
    if not os.path.exists('templates'):
        os.makedirs('templates')
    print("Servidor Flask iniciado. Acesse: http://127.0.0.1:5000/")
    app.run(debug=True)