from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer


Bot = ChatBot(
    'DAN',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 

training_data_personal = open('training/train.txt').read().splitlines()

trainer = ListTrainer(Bot)
trainer.train(training_data_personal)
  
trainer_corpus = ChatterBotCorpusTrainer(Bot)

trainer_corpus.train(
    'chatterbot.corpus.english'
)