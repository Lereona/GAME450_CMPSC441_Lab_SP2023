from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
#part 1 and 2:
#sentences we like to encode
sentences = ["The best food is a balanced and varied diet that provides all the necessary nutrients for optimal health and well-being.","It's difficult to define a single 'best' food, as different foods provide different nutrients and benefits, and the best diet depends on individual needs and preferences."]

#sentences are encoded by calling model.encode()
embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
print(embedding_1)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
print(embedding_2)


#print the cosine similarity
print(util.pytorch_cos_sim(embedding_1, embedding_2))

#part 3, our method:
#encode our 2 responses
response_1 = ["Determining the 'best' food is subjective and depends on a variety of factors, including individual dietary needs, personal preferences, and cultural traditions.",
"Generally, a healthy and balanced diet includes a variety of foods from all food groups, such as fruits, vegetables, whole grains, lean proteins, and healthy fats.",
"Consuming a variety of nutrient-dense foods can provide the necessary vitamins, minerals, and other nutrients needed for optimal health and well-being.",
"However, it's important to consider individual dietary needs and limitations, such as food allergies or intolerances, and consult with a healthcare professional or registered dietitian to determine the best diet for your specific needs."
]
encoded_response_1 = model.encode(response_1,convert_to_tensor=True)

response_2 = ["Determining the 'best' food can be a complex issue, as the optimal diet varies based on individual needs and preferences.",
"However, a balanced and varied diet that includes a wide range of nutrient-dense foods is generally considered to be the most beneficial for overall health and well-being.",
"Such a diet should include plenty of fruits, vegetables, whole grains, lean protein sources, and healthy fats, while limiting processed foods, sugar, and unhealthy fats.",
"Additionally, factors such as cultural background, food allergies or sensitivities, and personal ethics can also play a role in determining the best food for a particular individual.",
"Therefore, it's important to consult with a healthcare professional or registered dietitian to develop a personalized nutrition plan."
]
encoded_response_2 = model.encode(response_2,convert_to_tensor=True)

#loop through each sentence in response 1 and compare cosine similarity between each sentence in response 2. store the max similarities in a nested list
cos_sim_response_1 = []
cos_sim_response_2 = []

for i in range(4):
    best_sim = 0
    for j in range(5):
        temp_sim = util.pytorch_cos_sim(encoded_response_1[i], encoded_response_2[j]).tolist()
        if temp_sim[0][0] > best_sim:
            best_sim = temp_sim[0][0]
    
    cos_sim_response_1.append(best_sim)
#do the same but the other way around. this time compare each sentence in response 2 with each sentence in response 1
for i in range(5):
    best_sim = 0
    for j in range(4):
        temp_sim = util.pytorch_cos_sim(encoded_response_2[i], encoded_response_1[j]).tolist()
        if temp_sim[0][0] > best_sim:
            best_sim = temp_sim[0][0]
    
    cos_sim_response_2.append(best_sim)

#calculate average cosine similarity for each list. then, find average of the 2 list.
average_1 = sum(cos_sim_response_1)/len(cos_sim_response_1)
average_2 = sum(cos_sim_response_2)/len(cos_sim_response_2)
overall_average = (average_1 + average_2)/2

#this is the final metric to measure the similarity between the two responses
print(overall_average)



#prompts used and its responses:
#question 1-2
"Give me one sentence to explain what is the best food?"
#answer 1
"The best food is a balanced and varied diet that provides all the necessary nutrients for optimal health and well-being."
#answer 2
"It's difficult to define a single 'best' food, as different foods provide different nutrients and benefits, and the best diet depends on individual needs and preferences."

#question 3
"Give me one paragraph to explain what is the best food?"

#answer 1
"Determining the 'best' food is subjective and depends on a variety of factors, including individual dietary needs, personal preferences, and cultural traditions."
"Generally, a healthy and balanced diet includes a variety of foods from all food groups, such as fruits, vegetables, whole grains, lean proteins, and healthy fats."
"Consuming a variety of nutrient-dense foods can provide the necessary vitamins, minerals, and other nutrients needed for optimal health and well-being."
"However, it's important to consider individual dietary needs and limitations, such as food allergies or intolerances, and consult with a healthcare professional or registered dietitian to determine the best diet for your specific needs."

#answer 2
"Determining the 'best' food can be a complex issue, as the optimal diet varies based on individual needs and preferences."
"However, a balanced and varied diet that includes a wide range of nutrient-dense foods is generally considered to be the most beneficial for overall health and well-being."
"Such a diet should include plenty of fruits, vegetables, whole grains, lean protein sources, and healthy fats, while limiting processed foods, sugar, and unhealthy fats."
"Additionally, factors such as cultural background, food allergies or sensitivities, and personal ethics can also play a role in determining the best food for a particular individual."
"Therefore, it's important to consult with a healthcare professional or registered dietitian to develop a personalized nutrition plan."









