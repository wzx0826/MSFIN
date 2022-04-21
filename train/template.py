def set_template(args):
    # Set the templates here
  
    if args.template.find('MSFIN') >= 0:
        args.model = 'MSFIN'
        args.num_steps = 1
        args.n_feats = 24#20
        args.chop = True 
        
    if args.template.find('MSFIN_S') >= 0:
        args.model = 'MSFIN_S'
        args.num_steps = 0
        args.n_feats = 16#20
        args.chop = True
        