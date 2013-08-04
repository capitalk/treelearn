

class BaseTree(object):

  def process_work_items(self):
    max_pos = max(item.pos for item in work_items)
    if max_pos > len(self.values):
      self.grow_tree()
    
    # work items that need to create leaf nodes on the GPU
    leaves = []

    # work items that need to be fully grown into small subtrees per thradblock
    subtrees = []
    
    # work items that need a single split but are small enough to be loaded into shared memory
    small_splits = []
    
    # work items that have enough features to justify 
    # each one getting its own thread block
    block_per_feature = []

    # otherwise, launch a kernel for each feature
    kernel_per_feature = []

    for item in work_items:
      if item.nelts == 1 or item.purity == 1.9:
        leaves.append(item)
      elif item.nelts <= 32:
        subtrees.append(item)
      elif items.nelts * items.n_features * self.values.itemsize <= 4096:
        small_splits.append(item)
      elif item.n_features > 30:
        block_per_feature.append(item)
      else:
        kernel_per_feature.append(item)
        
        
        
    
