"""Regenerate Week 3 figures from frozen metrics and local completed artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
record=json.loads((ROOT/'experiments/independent_volume_evaluation/tv_reference_frozen.json').read_text())
rows=record['rows']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#2468a2','#d66b22']
fig,ax=plt.subplots(1,3,figsize=(10.4,2.8))
for i,start in enumerate([0,90]):
 vals=sorted([r for r in rows if r['case_id']==f'chest_start{start}_span120'],key=lambda r:r['lambda_tv'])
 for a,key,title in zip(ax,['psnr_peak_1','slice_aggregated_ssim','boundary_mae'],['Volume PSNR (dB) ↑','Slice-averaged SSIM ↑','Boundary MAE ↓']):
  a.plot([r['lambda_tv'] for r in vals],[r['metrics'][key] for r in vals],'-o',color=colors[i],label=f'Start {start}°')
  a.set_xscale('log',base=2);a.set_xticks([.025,.05,.1],['0.025','0.05','0.1']);a.set_xlabel('TV weight');a.set_title(title);a.axvline(.05,color='gray',ls=':',lw=1);a.grid(alpha=.2)
ax[0].legend();fig.tight_layout();fig.savefig(HERE/'figures/tv_tradeoff.pdf');plt.close(fig)
fig,axes=plt.subplots(2,4,figsize=(10,5.4))
for row,start in enumerate([0,90]):
 base=ROOT/f'output/limited_angle_generalization_v1/chest_start{start}_span120_r2_default_s0_i30000/point_cloud/iteration_30000'
 ref=np.load(base/'vol_gt.npy',mmap_mode='r')
 images=[np.take(ref,128,axis=2).T]
 for weight in [.025,.05,.1]:
  if weight==.05:path=base/'vol_pred.npy'
  else:
   suffix='__chani_20261008' if start==90 and weight==.1 else ''
   path=ROOT/f"output/ordinary_tv_controls/chest_start{start}_span120_ordinary_tv_{str(weight).replace('.', 'p')}_s0_i30000{suffix}/point_cloud/iteration_30000/vol_pred.npy"
  image=np.take(np.load(path,mmap_mode='r'),128,axis=2).T
  images.append(np.abs(image-images[0]))
 for col,im in enumerate(images):
  a=axes[row,col];handle=a.imshow(im,origin='lower',cmap='gray' if col==0 else 'magma',vmin=0,vmax=1 if col==0 else .15)
  a.set_xticks([]);a.set_yticks([])
  if row==0:a.set_title(['Reference','Error: TV 0.025','Error: TV 0.05','Error: TV 0.1'][col])
  if col==0:a.set_ylabel(f'Start {start}°',fontsize=11)
fig.subplots_adjust(left=.05,right=.91,top=.94,bottom=.04,wspace=.05,hspace=.08)
cax=fig.add_axes([.93,.18,.015,.64]);fig.colorbar(handle,cax=cax,label='Absolute error (saturated at 0.15)')
fig.savefig(HERE/'figures/slice_errors.png',dpi=220);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.4,2.8))
for ax,start in zip(axes,[0,90]):
 for weight,color in zip([.025,.05,.1],['#479c88','#2468a2','#d66b22']):
  if weight==.05:p=ROOT/f'output/limited_angle_generalization_v1/chest_start{start}_span120_r2_default_s0_i30000/eval'
  else:
   suffix='__chani_20261008' if start==90 and weight==.1 else ''
   p=ROOT/f"output/ordinary_tv_controls/chest_start{start}_span120_ordinary_tv_{str(weight).replace('.', 'p')}_s0_i30000{suffix}/eval"
  points=[]
  for f in sorted(p.glob('iter_*/eval3d.yml')):
   iteration=int(f.parent.name.split('_')[1])
   if iteration>=5000:points.append((iteration,yaml.safe_load(f.read_text())['psnr_3d']))
  ax.plot(*zip(*points),'-o',label=f'TV {weight}',color=color)
 ax.set_title(f'Acquisition start {start}°');ax.set_xlabel('Iteration');ax.set_ylabel('Volume PSNR (dB)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(HERE/'figures/convergence.pdf');plt.close(fig)
print('Generated three figures')
