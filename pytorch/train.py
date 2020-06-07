from pathlib import Path
from DatasetGenerator import DatasetGenerator
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from Densenet121 import DenseNet121
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve as prc, accuracy_score


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    data_path = '../dataset/cat_and_dog'
    train_dir = Path(f'{data_path}/train')
    valid_dir = Path(f'{data_path}/valid')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(contrast=(0.75, 1.25)),
        transforms.ToTensor(),
        # normalize with image_net stat
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    training_dataset = DatasetGenerator(path_dataset=train_dir, image_mode='L', transform=transform, device=device)
    train_loader = DataLoader(dataset=training_dataset, batch_size=8, num_workers=2, shuffle=True,
                              pin_memory=True)
    validate_dataset = DatasetGenerator(path_dataset=valid_dir, image_mode='L', transform=transform, device=device)
    validate_loader = DataLoader(dataset=training_dataset, batch_size=8, num_workers=2, shuffle=True,
                                 pin_memory=True)
    labels = training_dataset.classes
    count_classes = len(labels)
    model = DenseNet121(count_classes).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    loss_min = 1.0
    model_name = 'cat_dog_keras_pytorch'

    for epoch_id in range(1, 10):
        with torch.set_grad_enabled(True):
            print(f'Training Epoch {epoch_id}')
            loss_train = 0
            model.train()

            for batch_id, (input, target) in enumerate(train_loader):
                var_input = torch.autograd.Variable(input).to(device)
                var_target = torch.autograd.Variable(target).to(device)

                var_output = model(var_input).to(device)

                optimizer.zero_grad()

                loss_value = criterion(var_output, var_target)
                loss_value.backward()

                optimizer.step()
                scheduler.step(loss_value.data)
                loss_train += loss_value.data.item()

            loss_train = loss_train / len(train_loader)
            print('epoch', epoch_id, ' training loss:', loss_train)

        # validate
        with torch.no_grad():
            print(f'Validating Epoch {epoch_id}')
            model.eval()
            loss_validate = 0
            out_true = torch.FloatTensor().to(device)
            out_score = torch.FloatTensor().to(device)

            for batch_id, (input, target) in enumerate(validate_loader):
                var_input = torch.autograd.Variable(input).to(device)
                var_target = torch.autograd.Variable(target).to(device)

                var_output = model(var_input).to(device)

                loss_value = criterion(var_output, var_target).to(device)
                loss_validate += loss_value.data.item()

                out_true = torch.cat((out_true, var_target[:, 0]), 0)
                out_score = torch.cat((out_score, var_output[:, 0]), 0)

            auroc_mean = roc_auc_score(out_true, out_score, average='weighted')
            out_p, out_r, _ = prc(out_true, out_score)
            loss_validate = loss_validate / len(validate_loader)
            acc = ((out_score > 0.5) == out_true.byte()).float().mean().data.item()
            print('epoch', epoch_id, 'validate loss:', loss_validate, 'auroc', auroc_mean, 'acc', acc)

        if loss_validate < loss_min:
            scheduler.step(loss_validate)
            loss_min = loss_validate
            data_model = {
                'epoch': epoch_id + 1,
                'labels': labels,
                'state_dict': model.state_dict(),
                'best_loss': loss_min,
                'optimizer': optimizer.state_dict()
            }

            torch.save(data_model, model_name)
            print(
                f'Epoch[{epoch_id + 1}] [save] loss_train={str(loss_train)} , loss_validate={str(loss_validate)}')
            print(f'{model_name} Saved.')

        else:
            print(
                f'Epoch[{epoch_id + 1}] [----] loss={str(loss_train)}, loss_validate={str(loss_validate)}')
