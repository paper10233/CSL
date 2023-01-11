from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from tqdm import tqdm
from torch.autograd import Variable
from .min_norm_solvers import MinNormSolver, gradient_normalizers
import warnings
warnings.filterwarnings("ignore")


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    #print(self.model)
    self.loss = loss

  def forward(self, batch):
    #outputs = self.model(batch['input'])
    y_ppre = self.model.forward_tr(batch['ppre_input'])
    y_pre = self.model.forward_tr(batch['pre_input'])
    y = self.model.forward_tr(batch['input'])
    #outputs = self.model.head_with_align(y,pre_y)
    outputs_ppre,sappid,sappy = self.model.f_heads(y_ppre)
    outputs_pre,sapid,sapy = self.model.f_heads(y_pre)
    outputs,said,say = self.model.f_heads(y)

    #outputs = self.model.forward_tr1(batch['input'])
    loss_stats,loss = self.loss(outputs, batch, outputs_pre, outputs_ppre)
    return outputs[-1],loss_stats,loss

'''
class Model_base(torch.nn.Module):
  def __init__(self, model):
    super(Model_base, self).__init__()
    self.model = model

  def forward(self, batch):
    #outputs = self.model(batch['input'])
    y = self.model.forward_tr(batch['input'])
    
    return y


class Model_heads(torch.nn.Module):
  def __init__(self, model, loss):
    super(Model_heads, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, y,batch):

    outputs = self.model.f_heads(y)
    loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss_stats
'''

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.optimizer.add_param_group({'params': self.loss.parameters()})
    '''self.model_base = Model_base(model)
    self.model_heads = Model_heads(model,self.loss)    
    self.model={}
    self.model['base']=self.model_base
    self.model['heads']=self.model_heads'''
  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      '''self.model['base'] = DataParallel(
        self.model['base'], device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)
      self.model['heads'] = DataParallel(
        self.model['heads'], device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)'''
      self.model_with_loss= DataParallel(
        self.model_with_loss, device_ids=gpus,
        chunk_sizes=chunk_sizes).to(device)

    else:
      self.model_with_loss = self.model_with_loss.to(device)
      #self.model = self.model.to(device)


    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader, scale):

    model_with_loss = self.model_with_loss
    #model = self.model

    if phase == 'train':
      '''model['base'].train()
      model['heads'].train()'''
      model_with_loss.train()

    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      '''model['base'] = self.model['base'].module
        model['heads'] = self.model['heads'].module
      model['base'].eval()
      model['heads'].eval()'''

      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()

    for iter_id, batch in enumerate(data_loader):

      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)



      #loss = loss.mean()
      #ta = ['id_loss', 'det_loss']

      if phase == 'train':
        # ------------------------------------------------------
        '''
        grads = {}
        scale = {}
        loss_data = {}
        self.optimizer.zero_grad()
        y = model['base'](batch)
        y_variable = Variable(y.data.clone(), requires_grad=True)
        output, loss_stats = model['heads'](y_variable, batch)
        for t in ta:
          loss = loss_stats[t]
          loss = loss.mean()
          loss_data[t] = loss.data
          # print(t,' ',loss)
          loss.backward()
          grads[t] = []

          grads[t].append(Variable(y_variable.grad.data.clone(), requires_grad=False))
          # y_variable.grad.data.zero_()
          # print(t,' ',grads[t].type())

        # Normalize
        gn = gradient_normalizers(grads, loss_data, "loss+")
        for t in ta:
          for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # FW
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in ta])
        for i, t in enumerate(ta):
          scale[t] = float(sol[i])'''
        # -------------------------------------------------------------


        output, loss_stats, loss = model_with_loss(batch)
        '''y = model['base'](batch)
        output, loss_stats = model['heads'](y, batch)'''


        self.optimizer.zero_grad()
        '''for i,t in enumerate(ta):
          if i>0:
            loss=loss+scale[t]*loss_stats[t]
          else:
            loss=scale[t]*loss_stats[t]
        loss_stats['loss']=loss'''
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
      elif phase=='val':
        torch.cuda.empty_cache()
        with torch.no_grad():

          #output, loss_stats,loss = model_with_loss( batch)
          '''for i,t in enumerate(ta):
            if i>0:
              loss=loss+scale[t]*loss_stats[t]
            else:
              loss=scale[t]*loss_stats[t]
          loss_stats['loss']=loss'''

          #print(loss_stats)
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)


      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

      '''
      if epoch%1==0:
        for l in val_avg_loss_stats:
          val_avg_loss_stats[l].update(
            val_loss_stats[l].mean().item(), batch['input'].size(0))
          opt.print_iter = 100
          if iter_id % opt.print_iter == 0:
            print('val-------------------------')
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, val_avg_loss_stats[l].avg)'''

      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      
      if phase=='train':
        opt.print_iter=20
      else:
        #print(loss_stats)
        opt.print_iter=100
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()


      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats, batch



    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    '''val_ret={}
    if epoch%1==0:
      val_ret = {k: v.avg for k, v in val_avg_loss_stats.items()}
      val_ret['time'] = bar.elapsed_td.total_seconds() / 60.'''
      #return ret,results,val_ret
    #print(ret)
    return ret, results, scale

  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader,scale):
    return self.run_epoch('val', epoch, data_loader,scale)

  def train(self, epoch, data_loader,scale):
    return self.run_epoch('train', epoch, data_loader,scale)