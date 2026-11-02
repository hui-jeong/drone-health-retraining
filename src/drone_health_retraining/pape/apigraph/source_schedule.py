"""Source snapshot contract for the later integration team; no Test runner."""
def planned_source_context(target_year,*,selection_frozen=False,previous_year_complete=False):
    if target_year not in [2015,2016,2017,2018]:raise ValueError('Unsupported target year')
    if target_year>=2016 and not selection_frozen:raise ValueError('Freeze 2015 Validation selection before releasing 2015 labels')
    if target_year>=2017 and not previous_year_complete:raise ValueError('Previous Test year must be complete before adding its labels')
    return {'source_period':['2013-01',f'{target_year-1}-12'],'source_available_from_month':f'{target_year}-01',
      'source_version':f'source2013_{target_year-1}','target_year':target_year,'update_frequency':'annual',
      'label_availability_assumption':'all_previous_year_labels_available_at_next_january','execution_owner':'future_integration' if target_year>=2016 else 'pape_validation'}
