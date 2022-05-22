'''
This module is needed to get work genre categorical features
'''


def get_tag_ids(genre_infos: dict, get_label=False):
    tag_ids = set()
    for tag_list in genre_infos.values():
        for tag_value in tag_list:
          if get_label:
              tag_ids.add(tag_value[1]) 
          else: 
              tag_ids.add(tag_value[0])
    return tag_ids


'''
 We imply that this function computing 
 genre tag weights is suitable to represent them.
 'percent' is a number from 0 to 1 calculated by Fantlab
 as a percentage of the total number of voters 
 for this genre in the category.
'''
def make_tag_weight(vote_count, percent, num_voters):
    return min(vote_count * percent / num_voters, 1)


def item_features_buildup(genre_infos: dict, num_voters=20, use_label=False):
    """
    Returns iterable suitable for LightFM build_item_features with tag ids 
    as feature ids or text tag labels if use_label=True
    """
    tag_ids = get_tag_ids(genre_infos, use_label)
    item_features = []
    for key in genre_infos:
      for item in genre_infos[key]:
          weight = make_tag_weight(item[2], item[3], num_voters)
          if use_label:
              item_features.append((key, {item[1]: weight}))
          else:
             item_features.append((key, {item[0]: weight}))
    return item_features, tag_ids


def filter_by_work_id(item_features_raw: list, work_ids: list):
    return [(work_id, tag_id, tag, vote_count, percent) 
            for (work_id, tag_id, tag, vote_count, percent) in item_features_raw
            if work_id in work_ids]

def get_tag_ids_(item_features_raw: list, use_label=False) -> list:
    '''
    Returns label or id of a genre
    '''
    if use_label:
        tag_ids = set([tag for (work_id, tag_id, tag, vote_count, percent)
            in item_features_raw])
    else:
        tag_ids = set([tag_id for (work_id, tag_id, tag, vote_count, percent)
          in item_features_raw])
    return list(tag_ids)


def get_item_feature_weights(
  item_features_raw: list,
  use_label=False,
  num_voters=20
  ):
    '''
    Returns weights for genre features for every work
    '''
    if use_label:
       return [(work_id, {tag: make_tag_weight(vote_count, percent, num_voters)})
          for (work_id, tag_id, tag, vote_count, percent) in item_features_raw]

    else:
        return [(work_id, {tag_id: make_tag_weight(vote_count, percent, num_voters)})
          for (work_id, tag_id, tag, vote_count, percent) in item_features_raw]
    
