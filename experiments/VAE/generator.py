from tqdm import tqdm
from datetime import datetime
import tensorboardX, math, os
from losses import DistanceBasedLoss, SiameseLoss, ContrastiveLoss, SSIM
from discriminator import SiameseDiscriminator

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class SiameseGanSolver(object):
    """Solving GAN-like neural network with siamese discriminator."""

    def __init__(self, config, data_loader):
        """Set parameters of neural network and its training."""
        self.ssim_loss = SSIM()
        self.generator = config.generator
        self.discriminator = None
        self.distance_based_loss = None

        self.g_optimizer = None
        self.d_optimizer = None

        self.g_conv_dim = 128

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.learning_rate = 0.0001
        self.image_size = config.image_size
        self.num_epochs = config.num_epochs
        self.distance_weight = config.distance_weight

        self.data_loader = data_loader
#         print(self.data_loader.dataset)
        self.generate_path = config.generate_path
        self.model_path = config.model_path
        self.tensorboard = config.tensorboard
        self.device = config.device

        if self.tensorboard:
            self.tb_writer = tensorboardX.SummaryWriter(
                filename_suffix='_%s_%s' % (config.distance_weight, config.dataset))
            self.tb_graph_added = False

        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
#         self.generator = Generator(self.g_conv_dim, noise=self.noise, residual=self.residual)
        self.discriminator = SiameseDiscriminator(self.image_size)
        self.distance_based_loss = DistanceBasedLoss(2.0)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.learning_rate, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), self.learning_rate, [self.beta1, self.beta2])

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.distance_based_loss.to(self.device)

    def train(self):
        """Train generator and discriminator in minimax game."""
        # Prepare tensorboard writer
        if self.tensorboard:
            step = 0
        
        print("We are training\n")

        for epoch in tqdm(range(self.num_epochs)):
            print(str(epoch) + " " + str(datetime.now()))
#             i = 0
            for label, images0, images1 in self.data_loader:
#                 i += 1
#                 print(i)
                images0 = to_variable(images0)
                images1 = to_variable(images1)
#                 print("label:", label)
                label = to_variable(label)
#                 print("We extracted samples")
                # Train discriminator to recognize identity of real images
                output0, output1 = self.discriminator(images0, images1)
                d_real_loss = self.distance_based_loss(output0, output1, label)
#                 print("We calculated loss")
                # Backpropagation
                self.distance_based_loss.zero_grad()
                self.discriminator.zero_grad()
                d_real_loss.backward()
                self.d_optimizer.step()
#                 print("We did backprop")
                # Train discriminator to recognize identity of fake(privatized) images
                
                privatized_imgs, _, _ = self.generator(images0)
#                 print(privatized_imgs)
                output0, output1 = self.discriminator(images0, privatized_imgs)

                # Discriminator wants to minimize Euclidean distance between
                # original & privatized versions, hence label = 0
                d_fake_loss = self.distance_based_loss(output0, output1, 0)
                distance = 1.0 - self.ssim_loss(privatized_imgs, images0)
                d_fake_loss += self.distance_weight * distance
#                 print("We calculated loss")
                # Backpropagation
                self.distance_based_loss.zero_grad()
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                d_fake_loss.backward()
                self.d_optimizer.step()

                # Train generator to fool discriminator
                # Generator wants to push the distance between original & privatized
                # right to the margin, hence label = 1
                privatized_imgs, _, _ = self.generator(images0)
                output0, output1 = self.discriminator(images0, privatized_imgs)
                g_loss = self.distance_based_loss(output0, output1, 1)
                distance = 1.0 - self.ssim_loss(privatized_imgs, images0)
                g_loss += self.distance_weight * distance
#                 print("We calculated loss")
                # Backpropagation
                self.distance_based_loss.zero_grad()
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Write losses to tensorboard
                if self.tensorboard:
                    self.tb_writer.add_scalar('phase0/discriminator_real_loss',
                                              d_real_loss.item(), step)
                    self.tb_writer.add_scalar('phase0/discriminator_fake_loss',
                                              d_fake_loss.item(), step)
                    self.tb_writer.add_scalar('phase0/generator_loss',
                                              g_loss.item(), step)
                    self.tb_writer.add_scalar('phase0/distance_loss',
                                              distance.item(), step)

                    step += 1

            # Monitor training after each epoch
            if self.tensorboard:
                self._monitor_phase_0(self.tb_writer, step)

            # At the end save generator and discriminator to files
            if (epoch + 1) % 10 == 0:
                g_path = os.path.join(self.model_path, 'G', 'G-%d.pt' % (epoch+1))
                torch.save(self.generator.state_dict(), g_path)
                d_path = os.path.join(self.model_path, 'D', 'D-%d.pt' % (epoch+1))
                torch.save(self.discriminator.state_dict(), d_path)

        if self.tensorboard:
            self.tb_writer.close()

    def _monitor_phase_0(self, writer, step, n_images=10):
        # Measure accuracy of identity verification by discriminator
        correct_pairs = 0
        total_pairs = 0

        for label, images0, images1 in self.data_loader:
            images0 = to_variable(images0)
            images1 = to_variable(images1)
            label = to_variable(label)

            # Predict label = 1 if outputs are dissimilar (distance > margin)
            privatized_images0, _, _ = self.generator(images0)
            output0, output1 = self.discriminator(privatized_images0, images1)
            predictions = self.distance_based_loss.predict(output0, output1)
            predictions = predictions.type(label.data.type())

            correct_pairs += (predictions == label).sum().item()
            total_pairs += len(predictions == label)

            if total_pairs > 1000:
                break

        # Write accuracy to tensorboard
        accuracy = correct_pairs / total_pairs
        writer.add_scalar('phase0/discriminator_accuracy', accuracy, step)

        # Generate previews of privatized images
        reals, fakes = [], []
        for _, image, _ in self.data_loader.dataset:
#             print("i: ", image.shape)
            g_image, _, _ = self.generator(to_variable(image).unsqueeze(0))
            g_image = g_image.squeeze(0)
#             print("g: ", g_image.shape)
            reals.append(denorm(to_variable(image).data[0]))
            fakes.append(denorm(to_variable(g_image).data[0]))
            if len(reals) == n_images:
                break

        # Write images to tensorboard
        real_previews = torchvision.utils.make_grid(reals, nrow=n_images)
        fake_previews = torchvision.utils.make_grid(fakes, nrow=n_images)
#         print(real_previews.shape)
#         print(fake_previews.shape)
#         img = torchvision.utils.make_grid([real_previews, fake_previews], nrow=1)
        img = torchvision.utils.make_grid([*real_previews.unsqueeze_(1).unbind(0), *fake_previews.unsqueeze_(1).unbind(0)], nrow=10)
        writer.add_image('Previews', img, step)

    def generate(self):
        """Generate privatized images."""
        # Load trained parameters (generator)
        g_path = os.path.join(self.model_path, 'G', 'G-%d.pkl' % self.num_epochs)
        self.generator.load_state_dict(torch.load(g_path))
        self.generator.eval()

        # Generate the images
        for relative_path, image in self.data_loader:
            fake_image, _, _ = self.generator(to_variable(image))
            fake_path = os.path.join(self.generate_path, relative_path[0])
            if not os.path.exists(os.path.dirname(fake_path)):
                os.makedirs(os.path.dirname(fake_path))
            torchvision.utils.save_image(fake_image.data, fake_path, nrow=1)

    def check_discriminator_accuracy(self):
        """Measure discriminator's accuracy."""
        # Measure accuracy of identity verification by discriminator
        correct_pairs = 0
        total_pairs = 0

        g_path = os.path.join(self.model_path, 'G', 'G-%d.pkl' % self.num_epochs)
        self.generator.load_state_dict(torch.load(g_path))
        self.generator.eval()

        d_path = os.path.join(self.model_path, 'D', 'D-%d.pkl' % self.num_epochs)
        self.discriminator.load_state_dict(torch.load(d_path))
        self.discriminator.eval()

        for label, images0, images1 in self.data_loader:
            images0 = to_variable(images0)
            images1 = to_variable(images1)
            label = to_variable(label)

            # Predict label = 1 if outputs are dissimilar (distance > margin)
            privatized_images0, _, _ = self.generator(images0)
            output0, output1 = self.discriminator(privatized_images0, images1)
            predictions = self.distance_based_loss.predict(output0, output1)
            predictions = predictions.type(label.data.type())

            correct_pairs += (predictions == label).sum().item()
            total_pairs += len(predictions)

        accuracy = correct_pairs / total_pairs
        print('distance weight = %f' % self.distance_weight)
        print('accuracy = %f' % accuracy)
        
def to_variable(tensor):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def denorm(image):
    """Convert image range (-1, 1) to (0, 1)."""
    out = (image + 1) / 2
    return out.clamp(0, 1)