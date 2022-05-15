def get_tag_ids(genre_infos: dict, get_label=False):
    tag_ids = set()
    for tag_list in genre_infos.values():
        for tag_value in tag_list:
          if get_label==True:
              tag_ids.add(tag_value[1]) 
          else: 
              tag_ids.add(tag_value[0])
    return tag_ids

def make_tag_weight(vote_count, percent, num_voters):
    return min(vote_count * percent / num_voters, 1)

def item_features_buildup(genre_infos: dict, num_voters=20, use_label=False):
    '''
    Returns iterable suitable for LightFM build_item_features with tag ids 
    as feature ids or text tag labels if use_label=True
    '''
    tag_ids = get_tag_ids(genre_infos, use_label)
    item_features = []
    for key in genre_infos:
      for item in genre_infos[key]:
          weight = make_tag_weight(item[2], item[3], num_voters)
          if use_label==True:
              item_features.append((key, {item[1]: weight}))
          else:
             item_features.append((key, {item[0]: weight}))
    return item_features, tag_ids