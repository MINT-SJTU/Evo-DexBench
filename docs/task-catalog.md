# Task catalog

Evo-DexBench contains 30 tasks: 15 single-hand tasks (`m01_v1`–`m15_v1`) and 15 bimanual tasks (`b01_v1`–`b15_v1`). Task IDs are stable machine identifiers; task names and affordance families follow the paper taxonomy.

StageScore is the highest configured score whose stage predicate has been reached during an episode. Tasks may use ordered or independent stage predicates as declared in their packaged YAML. Success is the final predicate and is reported as SR over the evaluation set.

Use `dexbench tasks show TASK_ID` to inspect the task name, embodiment, scene, layouts, maximum episode steps, instruction, and stage definition.

## Button-driven

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `m01_v1` | `LampButton` | Press the lamp button to keep the lamp illuminated. | `button_contacted` (0.30); `button_triggered` (0.60); `lamp_stably_lit` (1.00) |
| `m02_v1` | `WallSwitch` | Flip the wall switch to the on position. | `reach_switch` (0.30); `half_press_switch` (0.60); `turn_on_switch` (1.00) |
| `m03_v1` | `PhoneHomeButton` | Press and release the red Home button on the phone lying on the table, so that the screen is displayed. | `home_button_contacted` (0.30); `home_button_pressed` (0.70); `home_button_released` (1.00) |
| `m04_v1` | `ToasterSlider` | Press down the toaster lever. | `reach_toaster` (0.30); `reach_slider` (0.60); `press_slider` (1.00) |
| `m05_v1` | `DisplayPowerButton` | Press the power button to turn on the display. | `power_button_proximity` (0.30); `power_button_contact` (0.60); `display_turned_on` (1.00) |

## Rotational manipulation

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `m06_v1` | `FaucetHandle` | Grasp the rotatable faucet handle and rotate it to the open position. | `faucet_handle_contacted` (0.30); `faucet_handle_midpoint_reached` (0.60); `faucet_opened` (1.00) |
| `m07_v1` | `MouseWheel` | Hold the mouse and scroll the wheel downward. | `mouse_grasped` (0.25); `wheel_contact` (0.50); `wheel_target_reached` (1.00) |
| `m08_v1` | `CoffeeKnob` | Turn the knob on the fixed coffee machine to setting two. | `coffee_knob_contacted` (0.30); `coffee_knob_rotated_correctly` (0.50); `coffee_knob_near_target` (0.75); `coffee_knob_aligned` (1.00) |
| `m09_v1` | `KettleLidTab` | Grasp the radial tab on top of the fixed kettle lid and rotate it until the end of the tab is aligned with the target notch on the kettle opening. | `cap_grasped` (0.30); `cap_positive_rotation` (0.55); `cap_target_neighborhood` (0.80); `cap_aligned_and_stable` (1.00) |
| `m10_v1` | `DishwasherKnob` | Turn the dishwasher knob to the target setting. | `dishwasher_knob_contacted` (0.30); `dishwasher_knob_near_target` (0.60); `dishwasher_knob_aligned` (1.00) |

## Access-and-transfer

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `m11_v1` | `CupFromCabinet` | Use the right hand to open the cabinet door, grasp the small cup by its handle, remove it, and place it fully on the countertop near the bottom of the cabinet. | `open_cabinet_door` (0.25); `grasp_jar` (0.50); `take_jar_out` (0.75); `place_jar_on_counter` (1.00) |
| `m12_v1` | `MicrowaveBowl` | Open the microwave door, place the bowl inside the microwave, and then close the microwave door. | `open_microwave_door` (0.25); `grasp_bowl` (0.50); `place_bowl_in_microwave` (0.75); `close_microwave_door` (1.00) |
| `m13_v1` | `TennisBallTransfer` | Open the sliding cabinet door, move the tennis ball from the first shelf to the second shelf and place it securely, then close the cabinet door. | `cabinet_accessible` (0.20); `tennis_ball_grasped` (0.45); `tennis_ball_stably_placed` (0.75); `cabinet_closed` (1.00) |
| `m14_v1` | `BananaFromDrawer` | Open the top drawer, take the banana out and place it securely on the desk, then close the drawer. | `drawer_accessible` (0.25); `banana_grasped` (0.40); `banana_outside_drawer` (0.60); `banana_stably_placed` (0.85); `drawer_closed` (1.00) |
| `m15_v1` | `CandleToCabinet` | Open any glass cabinet door above the dresser, place the candle upright from the lower shelf into the bottom level of the glass cabinet, release it. | `cabinet_door_opened` (0.25); `candle_grasped` (0.50); `candle_inside_cabinet` (0.75); `candle_stably_placed` (1.00) |

## Structured configuration

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `b01_v1` | `TablewarePlacement` | Arrange the tableware on the white tablecloth according to the reference image: place the plate in the center, the fork on the left, and the knife on the right. The fork and knife should be roughly aligned with the short edge of the table, with moderate skew allowed. | `one_tableware_placed` (0.30); `two_tableware_placed` (0.60); `all_tableware_placed` (0.90); `place_setting_stable` (1.00) |
| `b02_v1` | `HammerBlockAssembly` | Arrange the three blocks to match the reference hammer shape. | `one_block_correct` (0.25); `two_blocks_adjacent` (0.50); `three_blocks_correct` (0.75); `hammer_stably_complete` (1.00) |
| `b03_v1` | `SandwichAssembly` | Stack the bread and ingredients according to the reference image to make the specified sandwich. | `one_movable_ingredient_placed` (0.40); `both_movable_ingredients_placed` (0.80); `place_top_bread_and_stabilize` (1.00) |
| `b04_v1` | `RingTowerAssembly` | Following the tabletop diagram, place the three colored disks onto the post from largest to smallest. | `bottom_ring_on_post` (0.30); `middle_ring_on_post` (0.60); `tower_complete_and_stable` (1.00) |
| `b05_v1` | `BoxPacking` | Use both hands to place the three differently shaped objects into the box, then close the lid. | `first_object_packed` (0.20); `second_object_packed` (0.40); `all_objects_packed` (0.75); `box_lid_fully_closed` (1.00) |

## Assembly

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `b06_v1` | `PenCapping` | Lift the pen vertically from the fixed holder, hold the uncapped pen with one hand, align the cap with the pen tip using the other hand, and fully cap the pen. | `one_assembly_object_grasped` (0.20); `both_assembly_objects_grasped` (0.40); `cap_aligned_with_pen` (0.70); `pen_capped` (1.00) |
| `b07_v1` | `CupSleeveFit` | Hold the tapered cup with one hand, move the sleeve upward from the bottom with the other hand, and fit it around the middle of the cup. | `one_assembly_object_grasped` (0.20); `both_assembly_objects_grasped` (0.40); `sleeve_aligned_below_cup` (0.70); `sleeve_fitted_on_cup` (1.00) |
| `b08_v1` | `BottleFunnelInsertion` | Hold the hollow narrow-neck bottle with one hand, align the funnel with the bottle mouth using the other hand, and insert the funnel spout securely into the bottle. | `one_assembly_object_grasped` (0.20); `both_assembly_objects_grasped` (0.40); `funnel_aligned_with_bottle_mouth` (0.70); `funnel_inserted_in_bottle` (1.00) |
| `b09_v1` | `PlugInsertion` | Hold the socket and the two-pin plug with opposite hands, align both pins with the socket, and fully insert the plug. | `plug_and_socket_held` (0.25); `plug_aligned_with_socket` (0.50); `both_pins_entered_socket` (0.75); `plug_fully_inserted` (1.00) |
| `b10_v1` | `ScissorsSheathing` | Fully insert the tip of the closed scissors into the plastic scissors sheath. | `scissors_grasped` (0.20); `scissors_tip_aligned_with_sheath` (0.60); `scissors_fully_sheathed` (1.00) |

## Articulated configuration control

| ID | Task name | Instruction | Stages |
| --- | --- | --- | --- |
| `b11_v1` | `LaptopClosing` | Use one hand to stabilize the laptop base and the other hand to touch the screen, then close the screen to approximately 25 degrees. | `lid_edge_contacted` (0.30); `lid_mid_angle_reached` (0.60); `lid_target_angle_stable` (1.00) |
| `b12_v1` | `StaplerPaperMark` | Insert the sheet of paper between the stapler's upper and lower structures, press down to create a staple mark, and then move the stapler away. | `paper_grasped` (0.20); `paper_under_stapler` (0.50); `stapler_pressed` (0.70); `stapler_moved_after_mark` (1.00) |
| `b13_v1` | `ScissorsOpening` | Use one hand to pick up the scissors and the other hand to open them to their maximum opening angle. | `support_handle_stable` (0.40); `moving_handle_near_target` (0.70); `opening_angle_and_hold` (1.00) |
| `b14_v1` | `BoxLidOpening` | Use one hand to stabilize the box and the other hand to grasp the front edge or handle of the lid, then fully open the lid. | `box_stabilized` (0.30); `lid_opening_started` (0.60); `lid_target_angle_stable` (1.00) |
| `b15_v1` | `GlassesFolding` | Fold one temple of the glasses fully into the storage position, and fold the other temple at least to the stable closed position. | `left_temple_folded` (0.40); `right_temple_folded` (0.80); `both_temples_folded` (1.00) |
