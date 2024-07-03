import torch
from torch import optim, nn
from tqdm import tqdm
from sklearn import metrics
import pandas as pd


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, device, x_test):
    best_f1 = 1e-5
    best_ep = -1
    best_te_loss = 1e5
    f1_ls = []
    ans = x_test['label_index'].to_list()

    for epoch in range(epochs):
        print(f"Running epoch {epoch}, best val loss {best_te_loss} after epoch {best_ep}")
        model.train()
        tr_loss = 0
        step = 0
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            image, text, label = batch
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(text, image)
            loss = criterion(output, label)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            pbar.set_description(f"Train batchCE: {loss.item()}", refresh=True)
        scheduler.step()
        tr_loss /= step

        model.eval()
        te_loss = 0
        step = 0
        val_pbar = tqdm(val_dataloader, leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                step += 1
                image, text, label = batch
                image, label = image.to(device), label.to(device)
                output = model(text, image)
                loss = criterion(output, label)
                te_loss += loss.item()
                val_pbar.set_description(f"Val batchCE: {loss.item()}", refresh=True)
        te_loss /= step

        preds = []
        for batch in val_dataloader:
            image, text, label = batch
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                output = model(text, image)
                preds += output.argmax(dim=-1).tolist()

        f1_acc = metrics.f1_score(ans, preds, average='macro')
        print(f"Accuracy: {metrics.accuracy_score(ans, preds)}, F1: {f1_acc}")
        f1_ls.append(f1_acc)

        if best_f1 < f1_acc:
            best_f1 = f1_acc
            best_ep = epoch
            torch.save(model.state_dict(), "./KeylessAttAdamW.pt")

        print(f"Epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")

    torch.save(model.state_dict(), "./best-KeylessAttAdamW.pt")
