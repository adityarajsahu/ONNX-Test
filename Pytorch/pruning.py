import torch
import torchvision.models as models
import torch.nn.utils.prune as prune

model = models.resnet18(pretrained=True)

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.layer1[0].conv1, 'weight')
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)

prune.remove(model.conv1, 'weight')
prune.remove(model.layer1[0].conv1, 'weight')

for module, name in parameters_to_prune:
    print(f'Pruned Parameter: {name}')
    print(list(module.named_parameters(recurse=False))[0][1])

for name, param in model.named_parameters():
    if 'weight_orig' in name:
        print(f'Unpruned Parameter: {name}')
        print(param)

torch.save(model.state_dict(), 'Output/pruned_model.pt')
