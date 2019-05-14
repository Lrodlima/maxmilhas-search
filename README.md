# maxmilhas-search

Esse projeto faz parte de uma das etapas do processo seletivo da MaxMilhas. Toda o código aqui desenvolvido foi construído utilizando pySpark, python (3.6) e Apache Spark (2.3.4)


Para executar o projeto basta que você tenha o virtualenv instalado na sua máquina, instalar as dependências que estao no arquivo **requirements.txt** utilizado o pip.

A execucao do arquivo _data_preparation.py_ executa todo o pipeline de leitura e _cleansing_ do dataframe disponibilizado pela equipe da MaxMilhas. O final do processo, o dataset final é gravado no formato Parquet. Para executar o programa você deve informar o caminho do arquivo de entrada e o caminho dos arquivos de saída.


python data_preparation.py -dp '/path/to/input' -op '/path/to/output'