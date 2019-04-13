
import os
import time
import numpy as np
import torch
from sklearn.metrics.ranking import roc_auc_score

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, data_loader, criterion, optimizer, epoch,num_classes,competition, path,print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    results = []

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))

        y_true = target.detach().to('cpu').numpy().tolist()
        y_pred = output.detach().to('cpu').numpy().tolist()
        results.extend(list(zip(y_true, y_pred)))
        result = np.array(results)
        average_auc = calculate_auc(result[:, 0, :], result[:, 1, :], num_classes, competition)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Average AUC {acc:.3f}'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=average_auc))

        if i % (4800 - 1) == 0: # Checkpoint the model every 4800 iterations

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(path, "chkpt-{}.pth ".format(i)))

    return losses.avg, average_auc


def evaluate(model, device, data_loader, criterion, num_classes,competition, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    results = []
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            losses.update(loss.item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))
            result = np.array(results)
            average_auc = calculate_auc(result[:, 0, :], result[:, 1, :], num_classes, competition)
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Average AUC {acc:.3f}'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses, acc=average_auc))


    return losses.avg, average_auc, result


def calculate_auc(ground_truth, prediction, num_classes, competition=False):

    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.
    Index [2,5,6,8,10] correspond to the ChexPert competition labels : (a) Atelectasis,
    (b) Cardiomegaly, (c) Consolidation, (d) Edema, and (e) Pleural Effusion"""

    out_auroc = []

    index = [2,5,6,8,10] if competition else range(num_classes)

    for i in index:
        try:
            out_auroc.append(roc_auc_score(ground_truth[:, i], prediction[:, i]))
        except ValueError:
            pass

    return np.array(out_auroc,dtype=np.float32).mean()

