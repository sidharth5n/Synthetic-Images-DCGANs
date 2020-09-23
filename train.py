import torch
from torchvision import utils

# Find discriminator loss for real image
# Generate fake image and find discriminator loss
# Add the two discriminator loss and update discriminator
# Find discriminator loss on fake image
# Update generator.

def train(generator, discriminator, data_loader, loss_fn, optimizer_g, optimizer_d, device, epochs = 10):
    # for visualizing performance of generator
    fixed_noise = torch.randn(32, generator.feature_size, 1, 1, device = device)
    for epoch in range(epochs):
        total_loss_d = 0.0
        total_loss_g = 0.0
        for data, _ in data_loader:
            ############################
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            data = data.to(device)
            batch_size = data.size(0)
            label = torch.full(size = (batch_size,), fill_value = 1,
                               dtype = data.dtype, device = device)
            output = discriminator(data)
            loss_d_real = loss_fn(output, label)
            loss_d_real.backward()

            # train with fake
            noise = torch.randn(batch_size, generator.feature_size, 1, 1, device = device)
            fake = generator(noise)
            label.fill_(0)
            output = discriminator(fake.detach())
            loss_d_fake = loss_fn(output, label)
            loss_d_fake.backward()
            optimizer_d.step()

            ############################
            # Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            output = discriminator(fake)
            loss_g = loss_fn(output, label)
            loss_g.backward()
            optimizer_g.step()

            total_loss_d += loss_d_real.item() + loss_d_fake.item()
            total_loss_g += loss_g.item()

        print("Epoch {}/{} : Discriminator loss = {:.3f}, Generator loss = {:0.3f}".format(epoch + 1, epochs, total_loss_d / len(data_loader), total_loss_g/len(data_loader)))

        fake = generator(fixed_noise)
        utils.save_image(fake.detach(),
                         '/content/drive/My Drive/Colab Notebooks/DCGAN/fake_samples_epoch_{:02d}.png'.format(epoch + 1),
                         normalize=True)

        # save checkpoint
        torch.save(generator.state_dict(), 'checkpoint/generator_epoch_{:02d}.pth'.format(epoch + 1))
        torch.save(discriminator.state_dict(), 'checkpoint/generator_epoch_{:02d}.pth'.format(epoch + 1))
