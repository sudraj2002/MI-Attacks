from utils import *
import random

def train_vgg(model, optimizer, criterion, train_loader, num_epochs, device='cuda'): # defining train_model function to train the network
    model.train()
    attack_noise = create_attack('Noise', model=model, device=device, targeted=False)
    attack_fgsm = create_attack('FGSM', model=model, device=device, targeted=False)
    attack_pgd = create_attack('PGD', model=model, device=device, targeted=False)
    attack_cw = create_attack('CW', model=model, device=device, targeted=False)

    attacks = [attack_noise, attack_fgsm, attack_pgd, attack_cw]
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            labels = labels.to(device)

            if random.random() > 0.5:
                # Randomly attack the images 50% of the time
                attack = random.choice(attacks)
                images = attack.forward(images, labels)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    return model