def train_vgg(model, optimizer, criterion, train_loader, num_epochs, device='cuda'): # defining train_model function to train the network
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

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


def train_vit(model, classification_head, optimizer, criterion, train_loader, num_epochs, device='cuda'):
    model.to(device)
    classification_head.to(device)

    # Start training
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).last_hidden_state.mean(1)  # Average over sequence length and only the last but one layer
            logits = classification_head(outputs)  # Apply the head to the output features
            loss = criterion(logits, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training finished!")

    return model, classification_head

def train_my_vit(model, optimizer, criterion, train_loader, num_epochs, device='cuda'):
    model.to(device)

    # Start training
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()  # Backpropagate the loss
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print("Training finished!")

    return model