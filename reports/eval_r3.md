# r3

all       {"n": 40, "verdict_acc": 0.8, "recall": 0.71, "spec": 0.89, "time2": 0.4, "time4": 0.55, "type_acc": 0.45, "type_any": 0.55, "lat_p50": 60.5, "lat_max": 159.1, "score55": 32.8, "fp_types": {"no_machine_count": 1, "money_not_secured_before_next_client": 1}}
dev       {"n": 26, "verdict_acc": 0.77, "recall": 0.71, "spec": 0.83, "time2": 0.5, "time4": 0.67, "type_acc": 0.5, "type_any": 0.58, "lat_p50": 62.3, "lat_max": 159.1, "score55": 34.2, "fp_types": {"no_machine_count": 1, "money_not_secured_before_next_client": 1}}
holdout   {"n": 14, "verdict_acc": 0.86, "recall": 0.71, "spec": 1.0, "time2": 0.25, "time4": 0.38, "type_acc": 0.38, "type_any": 0.5, "lat_p50": 46.4, "lat_max": 106.3, "score55": 30.8, "fp_types": {}}
angle A   {"n": 26, "verdict_acc": 0.77, "recall": 0.69, "spec": 0.85, "time2": 0.38, "time4": 0.54, "type_acc": 0.54, "type_any": 0.54, "lat_p50": 59.8, "lat_max": 159.1, "score55": 33.1, "fp_types": {"no_machine_count": 1, "money_not_secured_before_next_client": 1}}
angle B   {"n": 10, "verdict_acc": 0.9, "recall": 0.8, "spec": 1.0, "time2": 0.5, "time4": 0.5, "type_acc": 0.5, "type_any": 1.0, "lat_p50": 70.4, "lat_max": 149.0, "score55": 37.5, "fp_types": {}}
angle C   {"n": 4, "verdict_acc": 0.75, "recall": 0.67, "spec": 1.0, "time2": 0.33, "time4": 0.67, "type_acc": 0.0, "type_any": 0.0, "lat_p50": 60.6, "lat_max": 66.9, "score55": 23.8, "fp_types": {}}

## По видео
OK  X-1 [dev/A] GT [('transfer_outside_procedure', 22)] | PRED [('no_machine_count', 20.5)] | 41.8s
ERR X-2 [dev/A] GT [('no_client_id_check', 10)] | PRED no_violation | 48.1s
OK  X-3 [dev/A] GT нет | PRED no_violation | 48.2s
OK  X-4 [holdout/A] GT [('money_not_secured_before_next_client', 57), ('no_machine_count', 68)] | PRED [('cashier_left_during_service', 46.0)] | 33.2s
OK  X-5 [holdout/A] GT нет | PRED no_violation | 26.0s
OK  X-11 [dev/B] GT [('devices_left_on', 18)] | PRED [('devices_left_on', 17.0)] | 62.3s
OK  X-12 [dev/B] GT нет | PRED no_violation | 116.2s
OK  X-13 [dev/B] GT [('no_machine_count', 47)] | PRED [('no_machine_count', 33.0)] | 149.0s
ERR X-14 [dev/A] GT нет | PRED [('no_machine_count', 33.0)] | 46.8s
OK  X-15 [holdout/A] GT [('valuables_left_on_leave', 15)] | PRED [('valuables_left_on_leave', 14.5)] | 60.7s
ERR X-22 [holdout/A] GT [('safe_key_left', 14)] | PRED no_violation | 61.6s
OK  X-23 [holdout/A] GT нет | PRED no_violation | 45.0s
OK  X-16 [dev/A] GT нет | PRED no_violation | 86.2s
OK  X-17 [dev/A] GT [('collector_bag_no_id_check', 7)] | PRED [('collector_bag_no_id_check', 7.0)] | 59.8s
OK  X-18 [dev/A] GT [('stranger_in_room', 20)] | PRED [('stranger_in_room', 17.0)] | 62.2s
ERR X-19 [dev/A] GT нет | PRED [('money_not_secured_before_next_client', 49.5)] | 121.3s
OK  X-20 [holdout/A] GT [('safe_open_during_work', 5)] | PRED [('safe_open_during_work', 7.5)] | 42.4s
OK  X-21 [holdout/A] GT нет | PRED no_violation | 53.7s
ERR X-24 [dev/A] GT [('valuables_left_on_leave', 8)] | PRED no_violation | 48.2s
OK  X-25 [dev/A] GT нет | PRED no_violation | 92.8s
OK  X-26 [dev/A] GT [('valuables_left_on_leave', 9)] | PRED [('valuables_left_on_leave', 10.0)] | 64.3s
OK  X-27 [dev/A] GT нет | PRED no_violation | 61.4s
OK  X-28 [holdout/A] GT [('laptop_not_locked', 7)] | PRED [('laptop_not_locked', 6.5)] | 45.3s
OK  X-29 [holdout/A] GT нет | PRED no_violation | 60.5s
OK  X-30 [dev/C] GT [('stranger_in_room', 13)] | PRED [('no_client_id_check', 13.5)] | 60.6s
OK  X-31 [dev/C] GT нет | PRED no_violation | 66.9s
ERR X-32 [dev/C] GT [('door_not_closed', 5)] | PRED no_violation | 50.9s
OK  X-33 [dev/C] GT [('door_not_closed', 6)] | PRED [('valuables_left_on_leave', 2.0)] | 44.0s
ERR X-34 [holdout/A] GT [('client_left_before_count', 13)] | PRED no_violation | 46.4s
OK  X-35 [holdout/A] GT нет | PRED no_violation | 106.3s
OK  X-37 [holdout/A] GT нет | PRED no_violation | 45.9s
OK  X-38 [dev/A] GT [('money_not_secured_before_next_client', 33)] | PRED [('money_not_secured_before_next_client', 31.5), ('no_client_id_check', 32.5)] | 143.7s
OK  X-39 [dev/A] GT нет | PRED no_violation | 159.1s
OK  X-40 [dev/B] GT [('personal_items', 17)] | PRED [('personal_items', 16.0)] | 71.3s
OK  X-8 [dev/B] GT нет | PRED no_violation | 88.5s
OK  X-43 [dev/B] GT нет | PRED no_violation | 54.7s
ERR X-44 [dev/B] GT [('no_machine_count', 16)] | PRED no_violation | 54.1s
OK  X-45 [dev/B] GT нет | PRED no_violation | 70.4s
OK  X-46 [holdout/B] GT [('food_drink', 11)] | PRED [('food_drink', 17.6)] | 56.2s
OK  X-10 [holdout/B] GT нет | PRED no_violation | 37.8s