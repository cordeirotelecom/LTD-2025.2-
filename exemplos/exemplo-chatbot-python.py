# Exemplo de Chatbot em Python

print('Olá! Sou um chatbot simples. Escreva "sair" para encerrar.')
while True:
    entrada = input('Você: ').lower()
    if 'oi' in entrada:
        print('Bot: Olá! Como posso ajudar?')
    elif 'tchau' in entrada:
        print('Bot: Até logo!')
    elif 'sair' in entrada:
        print('Bot: Encerrando...')
        break
    else:
        print('Bot: Não entendi.')
