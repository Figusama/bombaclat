# r5

all       {"n": 40, "verdict_acc": 0.82, "recall": 0.76, "spec": 0.89, "time2": 0.5, "time4": 0.65, "type_acc": 0.55, "type_any": 0.65, "lat_p50": 17.4, "lat_max": 68.7, "score55": 36.4, "fp_types": {"no_machine_count": 1, "client_left_before_count": 1}}
dev       {"n": 26, "verdict_acc": 0.77, "recall": 0.71, "spec": 0.83, "time2": 0.5, "time4": 0.67, "type_acc": 0.5, "type_any": 0.58, "lat_p50": 20.1, "lat_max": 68.7, "score55": 34.2, "fp_types": {"no_machine_count": 1, "client_left_before_count": 1}}
holdout   {"n": 14, "verdict_acc": 0.93, "recall": 0.86, "spec": 1.0, "time2": 0.5, "time4": 0.62, "type_acc": 0.62, "type_any": 0.75, "lat_p50": 16.3, "lat_max": 57.8, "score55": 40.1, "fp_types": {}}
angle A   {"n": 26, "verdict_acc": 0.85, "recall": 0.85, "spec": 0.85, "time2": 0.54, "time4": 0.77, "type_acc": 0.69, "type_any": 0.69, "lat_p50": 19.1, "lat_max": 68.7, "score55": 39.6, "fp_types": {"no_machine_count": 1, "client_left_before_count": 1}}
angle B   {"n": 10, "verdict_acc": 0.9, "recall": 0.8, "spec": 1.0, "time2": 0.5, "time4": 0.5, "type_acc": 0.5, "type_any": 1.0, "lat_p50": 13.8, "lat_max": 41.0, "score55": 37.5, "fp_types": {}}
angle C   {"n": 4, "verdict_acc": 0.5, "recall": 0.33, "spec": 1.0, "time2": 0.33, "time4": 0.33, "type_acc": 0.0, "type_any": 0.0, "lat_p50": 21.6, "lat_max": 21.9, "score55": 17.5, "fp_types": {}}

## По видео
OK  X-1 [dev/A] GT [('transfer_outside_procedure', 22)] | PRED [('no_machine_count', 20.5)] | 40.0s
ERR X-2 [dev/A] GT [('no_client_id_check', 10)] | PRED no_violation | 16.9s
OK  X-3 [dev/A] GT нет | PRED no_violation | 17.4s
OK  X-4 [holdout/A] GT [('money_not_secured_before_next_client', 57), ('no_machine_count', 68)] | PRED [('cashier_left_during_service', 46.0), ('money_not_secured_before_next_client', 56.5)] | 57.8s
OK  X-5 [holdout/A] GT нет | PRED no_violation | 26.2s
OK  X-11 [dev/B] GT [('devices_left_on', 18)] | PRED [('devices_left_on', 17.0)] | 17.3s
OK  X-12 [dev/B] GT нет | PRED no_violation | 26.1s
OK  X-13 [dev/B] GT [('no_machine_count', 47)] | PRED [('no_machine_count', 33.0)] | 41.0s
ERR X-14 [dev/A] GT нет | PRED [('no_machine_count', 33.0)] | 47.1s
OK  X-15 [holdout/A] GT [('valuables_left_on_leave', 15)] | PRED [('valuables_left_on_leave', 14.5)] | 16.9s
ERR X-22 [holdout/A] GT [('safe_key_left', 14)] | PRED no_violation | 17.2s
OK  X-23 [holdout/A] GT нет | PRED no_violation | 16.3s
OK  X-16 [dev/A] GT нет | PRED no_violation | 29.7s
OK  X-17 [dev/A] GT [('collector_bag_no_id_check', 7)] | PRED [('collector_bag_no_id_check', 7.0), ('client_left_before_count', 9.5)] | 27.9s
OK  X-18 [dev/A] GT [('stranger_in_room', 20)] | PRED [('stranger_in_room', 17.0)] | 18.8s
OK  X-19 [dev/A] GT нет | PRED no_violation | 56.0s
OK  X-20 [holdout/A] GT [('safe_open_during_work', 5)] | PRED [('safe_open_during_work', 7.5)] | 13.7s
OK  X-21 [holdout/A] GT нет | PRED no_violation | 10.2s
OK  X-24 [dev/A] GT [('valuables_left_on_leave', 8)] | PRED [('valuables_left_on_leave', 5.5)] | 19.1s
OK  X-25 [dev/A] GT нет | PRED no_violation | 21.1s
OK  X-26 [dev/A] GT [('valuables_left_on_leave', 9)] | PRED [('valuables_left_on_leave', 10.0)] | 20.1s
OK  X-27 [dev/A] GT нет | PRED no_violation | 17.1s
OK  X-28 [holdout/A] GT [('laptop_not_locked', 7)] | PRED [('laptop_not_locked', 6.5)] | 16.1s
OK  X-29 [holdout/A] GT нет | PRED no_violation | 16.2s
OK  X-30 [dev/C] GT [('stranger_in_room', 13)] | PRED [('no_client_id_check', 13.5)] | 21.9s
OK  X-31 [dev/C] GT нет | PRED no_violation | 15.2s
ERR X-32 [dev/C] GT [('door_not_closed', 5)] | PRED no_violation | 15.2s
ERR X-33 [dev/C] GT [('door_not_closed', 6)] | PRED no_violation | 21.6s
OK  X-34 [holdout/A] GT [('client_left_before_count', 13)] | PRED [('client_left_before_count', 12.5)] | 25.7s
OK  X-35 [holdout/A] GT нет | PRED no_violation | 16.2s
OK  X-37 [holdout/A] GT нет | PRED no_violation | 16.9s
OK  X-38 [dev/A] GT [('money_not_secured_before_next_client', 33)] | PRED [('client_left_before_count', 24.0), ('no_client_id_check', 32.5)] | 66.2s
ERR X-39 [dev/A] GT нет | PRED [('client_left_before_count', 38.5)] | 68.7s
OK  X-40 [dev/B] GT [('personal_items', 17)] | PRED [('personal_items', 16.0)] | 18.5s
OK  X-8 [dev/B] GT нет | PRED no_violation | 13.8s
OK  X-43 [dev/B] GT нет | PRED no_violation | 8.7s
ERR X-44 [dev/B] GT [('no_machine_count', 16)] | PRED no_violation | 8.4s
OK  X-45 [dev/B] GT нет | PRED no_violation | 10.0s
OK  X-46 [holdout/B] GT [('food_drink', 11)] | PRED [('food_drink', 17.6)] | 11.7s
OK  X-10 [holdout/B] GT нет | PRED no_violation | 7.9s