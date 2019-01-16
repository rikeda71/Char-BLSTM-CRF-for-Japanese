from deepjapaner.modelapi import ModelAPI

api = ModelAPI(model_path='model.pth', train_path='train.txt',
               wordemb_path='wordvectors', charemb_path='charvectors',
               hidden_size=300)

label = api.predict('私は六花亭の白い恋人を食べました')
print(label)
