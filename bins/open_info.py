from tensorboard.backend.event_processing import event_accumulator

ea=event_accumulator.EventAccumulator('./collect/dp_net_brain/logs/events.out.tfevents.1651458380.hbu.32622.0')
ea.Reload()
Tags = ea.Tags()  # 查看数据文件中的数据标签
print(Tags)
val_loss = ea.scalars.Items('val_loss')
print(val_loss)


