# -*- coding: utf-8 -*-

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Cargar el modelo y tokenizador
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Crear conjuntos de datos y dataloaders
train_texts = [
    "Como puedo contactar con el servicio de atencion al cliente",
    "Quiero realizar un seguimiento de mi pedido. Como puedo hacerlo",
    "Necesito ayuda con la configuracion de mi cuenta. Pueden asistirme",
    "Cuales son las opciones de envio y cuanto tiempo tomara la entrega",
    "Tengo un problema con un producto/servicio. Como puedo resolverlo",
    "Pueden proporcionarme informacion sobre las politicas de devolucion y reembolso",
    "No puedo acceder a mi cuenta. Como puedo restablecer mi contrasena",
    "Cual es el estado actual de mi solicitud o reclamo",
    "Quiero realizar cambios en mi pedido. Es posible",
    "Pueden proporcionarme informacion sobre promociones o descuentos actuales",
    "Come posso contattare il servizio clienti",
    "Voglio tracciare il mio ordine. Come posso farlo",
    "Ho bisogno di aiuto con la configurazione del mio account. Potete assistere",
    "Quali sono le opzioni di spedizione e quanto tempo ci vorra per la consegna",
    "Ho un problema con un prodotto/servizio. Come posso risolverlo",
    "Potete fornirmi informazioni sulle politiche di reso e rimborso",
    "Non riesco ad accedere al mio account. Come posso resettare la mia password",
    "Qual e lo stato attuale della mia richiesta o reclamo",
    "Voglio apportare modifiche al mio ordine. E possibile",
    "Potete fornirmi informazioni su promozioni o sconti attuali",
    "Comment puis-je contacter le service client",
    "Je veux suivre ma commande. Comment puis-je faire",
    "J'ai besoin d'aide pour la configuration de mon compte. Pouvez-vous m'aider",
    "Quelles sont les options d'expedition et combien de temps prendra la livraison",
    "J'ai un probleme avec un produit/service. Comment puis-je le resoudre",
    "Pouvez-vous me fournir des informations sur les politiques de retour et de remboursement",
    "Je ne peux pas acceder a mon compte. Comment puis-je reinitialiser mon mot de passe",
    "Quel est l'etat actuel de ma demande ou reclamation",
    "Je veux apporter des modifications a ma commande. Est-ce possible",
    "Pouvez-vous me fournir des informations sur les promotions ou les remises en cours",
    "How can I contact customer service",
    "I want to track my order. How can I do that",
    "I need help with the setup of my account. Can you assist",
    "What are the shipping options and how long will delivery take",
    "I have an issue with a product/service. How can I resolve it",
    "Can you provide information on return and refund policies",
    "I can't access my account. How can I reset my password",
    "What is the current status of my request or complaint",
    "I want to make changes to my order. Is it possible",
    "Can you provide information on current promotions or discounts",    
    "Hoe kan ik contact opnemen met de klantenservice",
    "Ik wil mijn bestelling volgen. Hoe kan ik dat doen",
    "Ik heb hulp nodig bij het instellen van mijn account. Kun je helpen",
    "Wat zijn de verzendopties en hoelang duurt de levering",
    "Ik heb een probleem met een product/dienst. Hoe kan ik dat oplossen",
    "Kun je informatie geven over retour- en terugbetalingsbeleid",
    "Ik kan niet bij mijn account. Hoe kan ik mijn wachtwoord resetten",
    "Wat is de huidige status van mijn verzoek of klacht",
    "Ik wil wijzigingen aanbrengen in mijn bestelling. Is dat mogelijk",
    "Kun je informatie geven over huidige promoties of kortingen",    
    "Wie kann ich den Kundenservice kontaktieren",
    "Ich mochte meine Bestellung verfolgen. Wie kann ich das tun",
    "Ich brauche Hilfe bei der Einrichtung meines Kontos. Konnen Sie helfen",
    "Welche Versandoptionen gibt es und wie lange dauert die Lieferung",
    "Ich habe ein Problem mit einem Produkt/Service. Wie kann ich es losen",
    "Konnen Sie Informationen zu Ruckgabe- und Ruckerstattungsrichtlinien geben",
    "Ich kann nicht auf mein Konto zugreifen. Wie kann ich mein Passwort zurucksetzen",
    "Was ist der aktuelle Status meiner Anfrage oder Beschwerde",
    "Ich mochte Anderungen an meiner Bestellung vornehmen. Ist das moglich",
    "Konnen Sie Informationen zu aktuellen Aktionen oder Rabatten geben",
    "What are the products/services you offer",
    "What are the prices of your products/services",
    "Do you have any current promotions or discounts",
    "What are the available payment options",
    "What is the delivery time for products/services",
    "Do you offer personalized services tailored to my needs",
    "How can I place an order or hire your services",
    "What are your return and refund policies",
    "Can you provide references from previous customers",
    "Do you offer post-sale customer support services",    
    "Quels sont les produits/services que vous offrez",
    "Quels sont les prix de vos produits/services",
    "Avez-vous des promotions ou des reductions en cours",
    "Quelles sont les options de paiement disponibles",
    "Quel est le delai de livraison pour les produits/services",
    "Proposez-vous des services personnalises adaptes a mes besoins",
    "Comment puis-je passer une commande ou utiliser vos services",
    "Quelles sont vos politiques de retour et de remboursement",
    "Pouvez-vous fournir des references de clients precedents",
    "Proposez-vous des services d'assistance client apres-vente",
    "Welche Produkte/Dienstleistungen bieten Sie an",
    "Wie sind die Preise Ihrer Produkte/Dienstleistungen",
    "Haben Sie derzeit Aktionen oder Rabatte",
    "Welche Zahlungsoptionen stehen zur Verfugung",
    "Wie lange betragt die Lieferzeit fur Produkte/Dienstleistungen",
    "Bieten Sie personalisierte Dienstleistungen an, die auf meine Bedurfnisse zugeschnitten sind",
    "Wie kann ich eine Bestellung aufgeben oder Ihre Dienstleistungen in Anspruch nehmen",
    "Was sind Ihre Ruckgabe- und Ruckerstattungsrichtlinien",
    "Konnen Sie Referenzen von fruheren Kunden bereitstellen",
    "Bieten Sie Kundensupport nach dem Kauf an",
    "Cuales son los productos/servicios que ofrecen",
    "Cuales son los precios de sus productos/servicios",
    "Tienen alguna promocion o descuento actual",
    "Cuales son las opciones de pago disponibles",
    "Cual es el plazo de entrega para los productos/servicios",
    "Ofrecen servicios personalizados o adaptados a mis necesidades",
    "Como puedo realizar un pedido o contratar sus servicios",
    "Cuales son sus politicas de devolucion y reembolso",
    "Pueden proporcionar referencias de clientes anteriores",
    "Ofrecen servicios de atencion al cliente postventa",
    "Quali sono i prodotti/servizi che offrite",
    "Quali sono i prezzi dei vostri prodotti/servizi",
    "Avete attualmente qualche promozione o sconto",
    "Quali sono le opzioni di pagamento disponibili",
    "Qual e il tempo di consegna per i prodotti/servizi",
    "Offrite servizi personalizzati o adattati alle mie esigenze",
    "Come posso effettuare un ordine o utilizzare i vostri servizi",
    "Quali sono le vostre politiche di reso e rimborso",
    "Potete fornire referenze da parte di clienti precedenti",
    "Offrite servizi di assistenza clienti post-vendita",
    "Wat zijn de producten/diensten die u aanbiedt",
    "Wat zijn de prijzen van uw producten/diensten",
    "Heeft u momenteel enige promotie of korting",
    "Wat zijn de beschikbare betalingsopties",
    "Wat is de levertijd voor de producten/diensten",
    "Biedt u gepersonaliseerde diensten aan die zijn afgestemd op mijn behoeften",
    "Hoe kan ik een bestelling plaatsen of uw diensten inhuren",
    "Wat zijn uw retour- en terugbetalingsbeleid",
    "Kunt u referenties verstrekken van eerdere klanten",
    "Biedt u klantenservice na verkoop aan"
]
# 0 para "duda comercial", 1 para "atención al cliente"
train_labels = [1] * 60 + [0] * 60

inputs = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True)
labels = torch.tensor(train_labels)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Configurar el optimizador y la función de pérdida
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Entrenamiento
model.train()
for epoch in range(3):  # número de épocas
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Guardar el modelo entrenado
model.save_pretrained("src/classifier/model/")
tokenizer.save_pretrained("src/classifier/model/")
