import torch
from torch import nn
import os
from pathlib import Path
import shutil

class HeadManager(nn.Module):
    def __init__(self):
        super(HeadManager, self).__init__()
        self.index_name_head = set()

    # end

    def register(self, head):
        name_head = head.__class__.__name__
        setattr(self, name_head, head)
        self.index_name_head.add(name_head)
        return self

    # end

    def forward(self, model, **kwargs):
        for name in self.index_name_head:
            head = getattr(self, name)
            head.forward(model, **kwargs)
        # end

    # end

    def get_head(self, klass):
        return getattr(self, klass.__name__)

    # end

    def clear_cache(self):
        for name_head in self.index_name_head:
            getattr(self, name_head).clear_cache()
        # end
    # end


# end


class Trainer(nn.Module):
    def __init__(self, model=None, manager=None):
        super(Trainer, self).__init__()
        self.model = model
        self.manager = manager

    # end

    def forward(self, **kwargs):
        self.clear_cache()

        self.model.forward(**kwargs)
        self.manager.forward(self.model, **kwargs)

    # end

    def clear_cache(self):
        self.model.clear_cache() if self.model else None
        self.manager.clear_cache() if self.manager else None
    # end


# end


class SaverAndLoader:
    def __init__(self, path_checkpoints='./checkpoints'):
        self.dict_name_item = {}
        self.path_checkpoints = path_checkpoints

    # end

    def add_item(self, item, name=None):
        if not name:
            name = item.__class__.__name__
        # end

        self.dict_name_item[name] = item
        return self

    # end

    def update_checkpoint(self, name_checkpoint, name_checkpoint_previous=None):  # epoch_n
        if not self.dict_name_item:
            print(f'[ALERT] no item added, skip saving checkpoint.')
            return
        # end

        if name_checkpoint_previous:
            result = self._delete_checkpoint_folder(name_checkpoint_previous)
            if result:
                print(f'[INFO] {name_checkpoint_previous} is cleared.')
            else:
                print(f'[ALERT] {name_checkpoint_previous} fail to be cleared.')
            # end
        # end

        folder_checkpoint = self._create_checkpoint_folder(name_checkpoint)
        for name_item, item in self.dict_name_item.items():
            path_checkpoint_item = os.path.join(folder_checkpoint, f'{name_item}.pt')
            torch.save(item.state_dict(), path_checkpoint_item)

            size_file_saved_MB = os.path.getsize(path_checkpoint_item) / 1024 / 1024
            print(f'[INFO] {name_item} is saved, {size_file_saved_MB} MB')
        # end

        print(f'[INFO] {name_checkpoint} is saved')

    # end

    def load_item_state(self, name_checkpoint, instance_item, name_item=None):
        if not name_item:
            name_item = instance_item.__class__.__name__
        # end

        path_checkpoint_item = os.path.join(self.path_checkpoints, name_checkpoint, f'{name_item}.pt')
        if not os.path.exists(path_checkpoint_item):
            print(f'[ERROR] {path_checkpoint_item} not exists')
            return None
        # end
        if issubclass(instance_item.__class__, torch.nn.Module):
            instance_item.load_state_dict(torch.load(path_checkpoint_item), strict=False)
        else:
            instance_item.load_state_dict(torch.load(path_checkpoint_item))
        # end

        print(f'[INFO] {name_item} loaded for {name_checkpoint}.')
        return instance_item

    # end

    def list_items(self):
        return list(self.dict_name_item.keys())

    # end

    def _create_checkpoint_folder(self, name_checkpoint):
        path_folder_target = os.path.join(self.path_checkpoints, name_checkpoint)
        Path(path_folder_target).mkdir(parents=True, exist_ok=True)
        return path_folder_target

    # end

    def _delete_checkpoint_folder(self, name_checkpoint_previous):
        path_folder_target = os.path.join(self.path_checkpoints, name_checkpoint_previous)
        if os.path.exists(path_folder_target):
            shutil.rmtree(path_folder_target, ignore_errors=True)
        # end
        return (not os.path.exists(path_folder_target))
    # end
# end