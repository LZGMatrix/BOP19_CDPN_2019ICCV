import os

# datasets = ['LMO', 'TLESS', 'YCBV', 'TUDL', 'HB', 'ICBIN', 'ITODD']
datasets = ['TUDL']
objects_dict = {
		'LMO': ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher'],
		'TLESS': [str(i) for i in range(1, 31)],
		# 'TLESS': [25],
		'YCBV': ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
           		'007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
           		'019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
           		'037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'],
		# 'TUDL': ['dragon', 'frog', 'can'],
		'TUDL': ['frog'],
		# 'HB': [str(i) for i in range(1, 34)],
		'HB': [str(i) for i in [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33]],
		'ICBIN': ['coffee_cup', 'juice_carton'],
		'ITODD': [str(i) for i in range(1, 29)]
		}
'''
# validation
for dataset in datasets:
	objects = objects_dict[dataset]
	for obj in objects:
		os.system('python main.py --cfg=cfg.yaml --exp_mod={} --dataset={} --object={}'.format('val', dataset, obj))
'''
# test
for dataset in datasets:
	objects = objects_dict[dataset]
	for obj in objects:
		os.system('python main.py --cfg=cfg.yaml --exp_mod={} --dataset={} --object={}'.format('test', dataset, obj))
		# os.system('python main.py --cfg=cfg.yaml --exp_mod={} --dataset={} --object={}'.format('val', dataset, obj))
