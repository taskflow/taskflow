(function () {
  'use strict';

  // simple data
  // A->B, A->C, B->D, and C->D
  const simple = [{"group":"executor 0","data":[{"label":"worker 4","data":[{"timeRange":[282,319],"name":"TaskC","val":"static task"},{"timeRange":[322,334],"name":"TaskD","val":"static task"}]},{"label":"worker 8","data":[{"timeRange":[125,269],"name":"TaskA","val":"static task"},{"timeRange":[273,283],"name":"TaskB","val":"static task"}]}]}];

  const kmeans=[{"group":"executor 0","data":[{"label":"worker 0","data":[{"timeRange":[28,276],"name":"allocate_c","val":"static task"},{"timeRange":[716,716],"name":"converged?","val":"condition task"},{"timeRange":[900,900],"name":"converged?","val":"condition task"},{"timeRange":[1049,1049],"name":"converged?","val":"condition task"},{"timeRange":[1097,1097],"name":"converged?","val":"condition task"},{"timeRange":[1406,1488],"name":"free","val":"static task"}]},{"label":"worker 4","data":[{"timeRange":[25,289],"name":"allocate_sy","val":"static task"},{"timeRange":[515,515],"name":"converged?","val":"condition task"},{"timeRange":[792,793],"name":"converged?","val":"condition task"},{"timeRange":[962,963],"name":"converged?","val":"condition task"},{"timeRange":[1166,1166],"name":"converged?","val":"condition task"},{"timeRange":[1237,1237],"name":"converged?","val":"condition task"}]},{"label":"worker 5","data":[{"timeRange":[15,192],"name":"allocate_my","val":"static task"}]},{"label":"worker 6","data":[{"timeRange":[21,206],"name":"allocate_sx","val":"static task"},{"timeRange":[640,640],"name":"converged?","val":"condition task"},{"timeRange":[1330,1330],"name":"converged?","val":"condition task"}]},{"label":"worker 8","data":[{"timeRange":[10,133],"name":"allocate_py","val":"static task"}]},{"label":"worker 9","data":[{"timeRange":[10,179],"name":"allocate_mx","val":"static task"}]},{"label":"worker 10","data":[{"timeRange":[7,112],"name":"allocate_px","val":"static task"}]},{"label":"worker 12","data":[{"timeRange":[199,359],"name":"h2d","val":"cudaflow"},{"timeRange":[359,505],"name":"update_means","val":"cudaflow"},{"timeRange":[537,614],"name":"update_means","val":"cudaflow"},{"timeRange":[641,712],"name":"update_means","val":"cudaflow"},{"timeRange":[717,782],"name":"update_means","val":"cudaflow"},{"timeRange":[823,888],"name":"update_means","val":"cudaflow"},{"timeRange":[903,957],"name":"update_means","val":"cudaflow"},{"timeRange":[969,1047],"name":"update_means","val":"cudaflow"},{"timeRange":[1062,1092],"name":"update_means","val":"cudaflow"},{"timeRange":[1103,1155],"name":"update_means","val":"cudaflow"},{"timeRange":[1169,1231],"name":"update_means","val":"cudaflow"},{"timeRange":[1239,1320],"name":"update_means","val":"cudaflow"},{"timeRange":[1335,1404],"name":"d2h","val":"cudaflow"}]}]}];

  const inference = [{"group":"executor 0","data":[{"label":"worker 72","data":[{"timeRange":[66,72],"name":"start","val":"static task"},{"timeRange":[76,4810],"name":"first_fetch","val":"condition task"},{"timeRange":[166928,166936],"name":"fetch","val":"condition task"},{"timeRange":[166937,166938],"name":"stop","val":"static task"}]},{"label":"worker 75","data":[{"timeRange":[80,12316],"name":"first_fetch","val":"condition task"},{"timeRange":[53138,55440],"name":"fetch","val":"condition task"},{"timeRange":[59840,63003],"name":"fetch","val":"condition task"},{"timeRange":[78671,82215],"name":"fetch","val":"condition task"},{"timeRange":[82255,85964],"name":"fetch","val":"condition task"},{"timeRange":[100783,104295],"name":"fetch","val":"condition task"},{"timeRange":[114624,117102],"name":"fetch","val":"condition task"},{"timeRange":[135122,138561],"name":"fetch","val":"condition task"},{"timeRange":[138584,142179],"name":"fetch","val":"condition task"},{"timeRange":[146330,146340],"name":"fetch","val":"condition task"},{"timeRange":[146340,146342],"name":"stop","val":"static task"},{"timeRange":[182331,182342],"name":"fetch","val":"condition task"},{"timeRange":[182342,182344],"name":"stop","val":"static task"},{"timeRange":[191204,191214],"name":"fetch","val":"condition task"},{"timeRange":[191215,191217],"name":"stop","val":"static task"}]},{"label":"worker 78","data":[{"timeRange":[183,12231],"name":"first_fetch","val":"condition task"}]},{"label":"worker 79","data":[{"timeRange":[247,12226],"name":"first_fetch","val":"condition task"}]},{"label":"worker 80","data":[{"timeRange":[12240,52941],"name":"GPU 0","val":"cudaflow"},{"timeRange":[55440,100668],"name":"GPU 0","val":"cudaflow"},{"timeRange":[104314,146104],"name":"GPU 0","val":"cudaflow"}]},{"label":"worker 81","data":[{"timeRange":[12512,82236],"name":"GPU 1","val":"cudaflow"},{"timeRange":[86008,138575],"name":"GPU 1","val":"cudaflow"},{"timeRange":[142226,182216],"name":"GPU 1","val":"cudaflow"}]},{"label":"worker 82","data":[{"timeRange":[12377,78571],"name":"GPU 2","val":"cudaflow"},{"timeRange":[82234,135011],"name":"GPU 2","val":"cudaflow"},{"timeRange":[138566,190859],"name":"GPU 2","val":"cudaflow"}]},{"label":"worker 83","data":[{"timeRange":[4927,59710],"name":"GPU 3","val":"cudaflow"},{"timeRange":[63006,114549],"name":"GPU 3","val":"cudaflow"},{"timeRange":[117105,166792],"name":"GPU 3","val":"cudaflow"}]}]}];

  const dreamplace=[{"group":"executor 0","data":[{"label":"worker 7","data":[{"timeRange":[41184,41680],"name":"7_0","val":"static task"}]},{"label":"worker 8","data":[{"timeRange":[41166,41891],"name":"8_0","val":"static task"},{"timeRange":[56741,57578],"name":"8_1","val":"static task"}]},{"label":"worker 9","data":[{"timeRange":[41136,42422],"name":"9_0","val":"static task"},{"timeRange":[56684,58470],"name":"9_1","val":"static task"}]},{"label":"worker 10","data":[{"timeRange":[41094,43122],"name":"10_0","val":"static task"},{"timeRange":[56537,61770],"name":"10_1","val":"static task"}]},{"label":"worker 11","data":[{"timeRange":[41064,44832],"name":"11_0","val":"static task"},{"timeRange":[56428,65340],"name":"11_1","val":"static task"},{"timeRange":[89724,89967],"name":"11_2","val":"static task"},{"timeRange":[97863,98208],"name":"11_3","val":"static task"}]},{"label":"worker 12","data":[{"timeRange":[41034,44790],"name":"12_0","val":"static task"},{"timeRange":[56611,58324],"name":"12_1","val":"static task"}]},{"label":"worker 13","data":[{"timeRange":[41016,48744],"name":"13_0","val":"static task"},{"timeRange":[56211,89349],"name":"13_1","val":"static task"},{"timeRange":[89350,89350],"name":"13_2","val":"static task"},{"timeRange":[89350,89351],"name":"13_3","val":"static task"},{"timeRange":[89369,89370],"name":"13_4","val":"static task"},{"timeRange":[89370,89370],"name":"13_5","val":"static task"},{"timeRange":[89371,89371],"name":"13_6","val":"static task"},{"timeRange":[89371,89371],"name":"13_7","val":"static task"},{"timeRange":[89371,89372],"name":"13_8","val":"static task"},{"timeRange":[89372,89372],"name":"13_9","val":"static task"},{"timeRange":[89373,89373],"name":"13_10","val":"static task"},{"timeRange":[89373,89373],"name":"13_11","val":"static task"},{"timeRange":[89373,89373],"name":"13_12","val":"static task"},{"timeRange":[89373,89373],"name":"13_13","val":"static task"},{"timeRange":[89374,89374],"name":"13_14","val":"static task"},{"timeRange":[89374,89374],"name":"13_15","val":"static task"},{"timeRange":[89374,89374],"name":"13_16","val":"static task"},{"timeRange":[89374,89374],"name":"13_17","val":"static task"},{"timeRange":[89374,89374],"name":"13_18","val":"static task"},{"timeRange":[89375,89375],"name":"13_19","val":"static task"},{"timeRange":[89375,89375],"name":"13_20","val":"static task"},{"timeRange":[89375,89375],"name":"13_21","val":"static task"},{"timeRange":[89375,89375],"name":"13_22","val":"static task"},{"timeRange":[89376,89376],"name":"13_23","val":"static task"},{"timeRange":[89376,89376],"name":"13_24","val":"static task"},{"timeRange":[89376,89376],"name":"13_25","val":"static task"},{"timeRange":[89376,89377],"name":"13_26","val":"static task"},{"timeRange":[89377,89377],"name":"13_27","val":"static task"},{"timeRange":[89377,89377],"name":"13_28","val":"static task"},{"timeRange":[89377,89377],"name":"13_29","val":"static task"},{"timeRange":[89377,89377],"name":"13_30","val":"static task"},{"timeRange":[89377,89378],"name":"13_31","val":"static task"},{"timeRange":[89378,89385],"name":"13_32","val":"static task"},{"timeRange":[89386,89386],"name":"13_33","val":"static task"},{"timeRange":[89386,89386],"name":"13_34","val":"static task"},{"timeRange":[89386,89386],"name":"13_35","val":"static task"},{"timeRange":[89387,89387],"name":"13_36","val":"static task"},{"timeRange":[89387,89387],"name":"13_37","val":"static task"},{"timeRange":[89387,89387],"name":"13_38","val":"static task"},{"timeRange":[89387,89387],"name":"13_39","val":"static task"},{"timeRange":[89388,89388],"name":"13_40","val":"static task"},{"timeRange":[89388,89388],"name":"13_41","val":"static task"},{"timeRange":[89388,89388],"name":"13_42","val":"static task"},{"timeRange":[89388,89388],"name":"13_43","val":"static task"},{"timeRange":[89389,89389],"name":"13_44","val":"static task"},{"timeRange":[89389,89389],"name":"13_45","val":"static task"},{"timeRange":[89389,89389],"name":"13_46","val":"static task"},{"timeRange":[89389,89389],"name":"13_47","val":"static task"},{"timeRange":[89390,89390],"name":"13_48","val":"static task"},{"timeRange":[89390,89391],"name":"13_49","val":"static task"},{"timeRange":[89391,89391],"name":"13_50","val":"static task"},{"timeRange":[89391,89391],"name":"13_51","val":"static task"},{"timeRange":[89391,89391],"name":"13_52","val":"static task"},{"timeRange":[89392,89392],"name":"13_53","val":"static task"},{"timeRange":[89392,89392],"name":"13_54","val":"static task"},{"timeRange":[89392,89392],"name":"13_55","val":"static task"},{"timeRange":[89392,89393],"name":"13_56","val":"static task"},{"timeRange":[89393,89393],"name":"13_57","val":"static task"},{"timeRange":[89393,89393],"name":"13_58","val":"static task"},{"timeRange":[89393,89394],"name":"13_59","val":"static task"},{"timeRange":[89394,89394],"name":"13_60","val":"static task"},{"timeRange":[89394,89394],"name":"13_61","val":"static task"},{"timeRange":[89394,89394],"name":"13_62","val":"static task"},{"timeRange":[89394,89394],"name":"13_63","val":"static task"},{"timeRange":[89394,89409],"name":"13_64","val":"static task"},{"timeRange":[89409,89409],"name":"13_65","val":"static task"},{"timeRange":[89410,89410],"name":"13_66","val":"static task"},{"timeRange":[89410,89410],"name":"13_67","val":"static task"},{"timeRange":[89410,89410],"name":"13_68","val":"static task"},{"timeRange":[89410,89410],"name":"13_69","val":"static task"},{"timeRange":[89411,89411],"name":"13_70","val":"static task"},{"timeRange":[89411,89411],"name":"13_71","val":"static task"},{"timeRange":[89411,89411],"name":"13_72","val":"static task"},{"timeRange":[89411,89411],"name":"13_73","val":"static task"},{"timeRange":[89412,89412],"name":"13_74","val":"static task"},{"timeRange":[89412,89412],"name":"13_75","val":"static task"},{"timeRange":[89412,89412],"name":"13_76","val":"static task"},{"timeRange":[89412,89412],"name":"13_77","val":"static task"},{"timeRange":[89413,89413],"name":"13_78","val":"static task"},{"timeRange":[89413,89413],"name":"13_79","val":"static task"},{"timeRange":[89413,89413],"name":"13_80","val":"static task"},{"timeRange":[89413,89413],"name":"13_81","val":"static task"},{"timeRange":[89414,89414],"name":"13_82","val":"static task"},{"timeRange":[89414,89414],"name":"13_83","val":"static task"},{"timeRange":[89414,89414],"name":"13_84","val":"static task"},{"timeRange":[89414,89414],"name":"13_85","val":"static task"},{"timeRange":[89414,89415],"name":"13_86","val":"static task"},{"timeRange":[89415,89415],"name":"13_87","val":"static task"},{"timeRange":[89415,89415],"name":"13_88","val":"static task"},{"timeRange":[89415,89415],"name":"13_89","val":"static task"},{"timeRange":[89416,89416],"name":"13_90","val":"static task"},{"timeRange":[89416,89416],"name":"13_91","val":"static task"},{"timeRange":[89416,89416],"name":"13_92","val":"static task"},{"timeRange":[89417,89417],"name":"13_93","val":"static task"},{"timeRange":[89417,89417],"name":"13_94","val":"static task"},{"timeRange":[89417,89417],"name":"13_95","val":"static task"},{"timeRange":[89417,89418],"name":"13_96","val":"static task"},{"timeRange":[89418,89418],"name":"13_97","val":"static task"},{"timeRange":[89418,89419],"name":"13_98","val":"static task"},{"timeRange":[89419,89419],"name":"13_99","val":"static task"},{"timeRange":[89419,89419],"name":"13_100","val":"static task"},{"timeRange":[89419,89419],"name":"13_101","val":"static task"},{"timeRange":[89420,89420],"name":"13_102","val":"static task"},{"timeRange":[89420,89420],"name":"13_103","val":"static task"},{"timeRange":[89420,89420],"name":"13_104","val":"static task"},{"timeRange":[89420,89420],"name":"13_105","val":"static task"},{"timeRange":[89421,89421],"name":"13_106","val":"static task"},{"timeRange":[89421,89421],"name":"13_107","val":"static task"},{"timeRange":[89421,89421],"name":"13_108","val":"static task"},{"timeRange":[89421,89421],"name":"13_109","val":"static task"},{"timeRange":[89421,89421],"name":"13_110","val":"static task"},{"timeRange":[89422,89422],"name":"13_111","val":"static task"},{"timeRange":[89422,89422],"name":"13_112","val":"static task"},{"timeRange":[89422,89422],"name":"13_113","val":"static task"},{"timeRange":[89423,89423],"name":"13_114","val":"static task"},{"timeRange":[89423,89423],"name":"13_115","val":"static task"},{"timeRange":[89423,89423],"name":"13_116","val":"static task"},{"timeRange":[89423,89423],"name":"13_117","val":"static task"},{"timeRange":[89423,89423],"name":"13_118","val":"static task"},{"timeRange":[89424,89424],"name":"13_119","val":"static task"},{"timeRange":[89424,89424],"name":"13_120","val":"static task"},{"timeRange":[89424,89424],"name":"13_121","val":"static task"},{"timeRange":[89425,89425],"name":"13_122","val":"static task"},{"timeRange":[89425,89425],"name":"13_123","val":"static task"},{"timeRange":[89425,89425],"name":"13_124","val":"static task"},{"timeRange":[89425,89425],"name":"13_125","val":"static task"},{"timeRange":[89426,89426],"name":"13_126","val":"static task"},{"timeRange":[89426,89426],"name":"13_127","val":"static task"},{"timeRange":[89426,89446],"name":"13_128","val":"static task"},{"timeRange":[89446,89446],"name":"13_129","val":"static task"},{"timeRange":[89446,89447],"name":"13_130","val":"static task"},{"timeRange":[89447,89447],"name":"13_131","val":"static task"},{"timeRange":[89447,89447],"name":"13_132","val":"static task"},{"timeRange":[89447,89447],"name":"13_133","val":"static task"},{"timeRange":[89447,89448],"name":"13_134","val":"static task"},{"timeRange":[89448,89448],"name":"13_135","val":"static task"},{"timeRange":[89448,89448],"name":"13_136","val":"static task"},{"timeRange":[89448,89449],"name":"13_137","val":"static task"},{"timeRange":[89449,89449],"name":"13_138","val":"static task"},{"timeRange":[89449,89449],"name":"13_139","val":"static task"},{"timeRange":[89449,89456],"name":"13_140","val":"static task"},{"timeRange":[89457,89462],"name":"13_141","val":"static task"},{"timeRange":[89462,89469],"name":"13_142","val":"static task"},{"timeRange":[89469,89481],"name":"13_143","val":"static task"},{"timeRange":[89481,89502],"name":"13_144","val":"static task"},{"timeRange":[89502,89536],"name":"13_145","val":"static task"},{"timeRange":[89537,89594],"name":"13_146","val":"static task"},{"timeRange":[89594,89688],"name":"13_147","val":"static task"},{"timeRange":[89688,89893],"name":"13_148","val":"static task"}]},{"label":"worker 14","data":[{"timeRange":[40989,52673],"name":"14_0","val":"static task"},{"timeRange":[55788,77870],"name":"14_1","val":"static task"},{"timeRange":[89414,89590],"name":"14_2","val":"static task"},{"timeRange":[89597,89898],"name":"14_3","val":"static task"}]},{"label":"worker 15","data":[{"timeRange":[40934,49025],"name":"15_0","val":"static task"},{"timeRange":[56288,67777],"name":"15_1","val":"static task"},{"timeRange":[89654,89975],"name":"15_2","val":"static task"},{"timeRange":[97806,98538],"name":"15_3","val":"static task"}]},{"label":"worker 16","data":[{"timeRange":[40963,53709],"name":"16_0","val":"static task"},{"timeRange":[55765,66073],"name":"16_1","val":"static task"},{"timeRange":[89705,89987],"name":"16_2","val":"static task"},{"timeRange":[98004,99131],"name":"16_3","val":"static task"}]},{"label":"worker 17","data":[{"timeRange":[40890,51849],"name":"17_0","val":"static task"},{"timeRange":[55861,64897],"name":"17_1","val":"static task"}]},{"label":"worker 18","data":[{"timeRange":[40922,50867],"name":"18_0","val":"static task"},{"timeRange":[55981,67107],"name":"18_1","val":"static task"},{"timeRange":[89726,89959],"name":"18_2","val":"static task"},{"timeRange":[98135,99804],"name":"18_3","val":"static task"}]},{"label":"worker 19","data":[{"timeRange":[40860,54230],"name":"19_0","val":"static task"},{"timeRange":[55654,67521],"name":"19_1","val":"static task"},{"timeRange":[89706,89963],"name":"19_2","val":"static task"}]},{"label":"worker 20","data":[{"timeRange":[40717,50280],"name":"20_0","val":"static task"},{"timeRange":[56064,68604],"name":"20_1","val":"static task"},{"timeRange":[89620,89916],"name":"20_2","val":"static task"}]},{"label":"worker 21","data":[{"timeRange":[40711,51375],"name":"21_0","val":"static task"},{"timeRange":[55883,69015],"name":"21_1","val":"static task"},{"timeRange":[89589,89832],"name":"21_2","val":"static task"}]},{"label":"worker 22","data":[{"timeRange":[40672,50347],"name":"22_0","val":"static task"},{"timeRange":[55988,64991],"name":"22_1","val":"static task"}]},{"label":"worker 23","data":[{"timeRange":[40630,40647],"name":"23_0","val":"static task"},{"timeRange":[40729,48294],"name":"23_1","val":"static task"},{"timeRange":[56351,61270],"name":"23_2","val":"static task"}]},{"label":"worker 24","data":[{"timeRange":[40698,53850],"name":"24_0","val":"static task"},{"timeRange":[55678,65707],"name":"24_1","val":"static task"}]},{"label":"worker 25","data":[{"timeRange":[40694,49433],"name":"25_0","val":"static task"},{"timeRange":[56196,67871],"name":"25_1","val":"static task"},{"timeRange":[89646,89950],"name":"25_2","val":"static task"}]},{"label":"worker 26","data":[{"timeRange":[176,4057],"name":"26_0","val":"static task"},{"timeRange":[40501,40510],"name":"26_1","val":"static task"},{"timeRange":[40515,40529],"name":"26_2","val":"static task"},{"timeRange":[40533,40545],"name":"26_3","val":"static task"},{"timeRange":[40549,40564],"name":"26_4","val":"static task"},{"timeRange":[40569,40587],"name":"26_5","val":"static task"},{"timeRange":[40588,40607],"name":"26_6","val":"static task"},{"timeRange":[40615,40631],"name":"26_7","val":"static task"},{"timeRange":[40673,48430],"name":"26_8","val":"static task"},{"timeRange":[56339,67209],"name":"26_9","val":"static task"},{"timeRange":[89666,89948],"name":"26_10","val":"static task"},{"timeRange":[97888,98838],"name":"26_11","val":"static task"}]},{"label":"worker 27","data":[{"timeRange":[40695,50196],"name":"27_0","val":"static task"},{"timeRange":[56082,67355],"name":"27_1","val":"static task"},{"timeRange":[89714,89966],"name":"27_2","val":"static task"},{"timeRange":[98080,99420],"name":"27_3","val":"static task"}]},{"label":"worker 28","data":[{"timeRange":[185,187],"name":"28_0","val":"static task"},{"timeRange":[187,12632],"name":"28_1","val":"static task"},{"timeRange":[12632,12633],"name":"28_2","val":"static task"},{"timeRange":[12633,12846],"name":"28_3","val":"static task"},{"timeRange":[13551,13551],"name":"28_4","val":"static task"},{"timeRange":[15625,15626],"name":"28_5","val":"static task"},{"timeRange":[15626,15627],"name":"28_6","val":"condition task"},{"timeRange":[15627,15627],"name":"28_7","val":"static task"},{"timeRange":[40530,40546],"name":"28_8","val":"static task"},{"timeRange":[40553,40567],"name":"28_9","val":"static task"},{"timeRange":[40573,40590],"name":"28_10","val":"static task"},{"timeRange":[40592,40604],"name":"28_11","val":"static task"},{"timeRange":[40620,40641],"name":"28_12","val":"static task"},{"timeRange":[40679,48793],"name":"28_13","val":"static task"},{"timeRange":[56301,67493],"name":"28_14","val":"static task"},{"timeRange":[89726,89997],"name":"28_15","val":"static task"},{"timeRange":[89998,89999],"name":"28_16","val":"static task"},{"timeRange":[89999,97788],"name":"28_17","val":"static task"},{"timeRange":[97790,97854],"name":"28_18","val":"static task"},{"timeRange":[97854,97869],"name":"28_19","val":"static task"},{"timeRange":[97870,97890],"name":"28_20","val":"static task"},{"timeRange":[97890,98271],"name":"28_21","val":"static task"}]},{"label":"worker 29","data":[{"timeRange":[172,2359],"name":"29_0","val":"static task"},{"timeRange":[40681,49912],"name":"29_1","val":"static task"},{"timeRange":[56098,77030],"name":"29_2","val":"static task"},{"timeRange":[89422,89527],"name":"29_3","val":"static task"},{"timeRange":[89531,89665],"name":"29_4","val":"static task"},{"timeRange":[89669,89907],"name":"29_5","val":"static task"}]},{"label":"worker 30","data":[{"timeRange":[177,3315],"name":"30_0","val":"static task"},{"timeRange":[40594,40608],"name":"30_1","val":"static task"},{"timeRange":[40621,40640],"name":"30_2","val":"static task"},{"timeRange":[40670,51319],"name":"30_3","val":"static task"},{"timeRange":[55900,70579],"name":"30_4","val":"static task"},{"timeRange":[89496,89628],"name":"30_5","val":"static task"},{"timeRange":[89633,89849],"name":"30_6","val":"static task"}]},{"label":"worker 31","data":[{"timeRange":[158,3974],"name":"31_0","val":"static task"},{"timeRange":[15055,15056],"name":"31_1","val":"static task"},{"timeRange":[15056,15057],"name":"31_2","val":"condition task"},{"timeRange":[15057,15058],"name":"31_3","val":"static task"},{"timeRange":[16147,16147],"name":"31_4","val":"static task"},{"timeRange":[16148,16148],"name":"31_5","val":"condition task"},{"timeRange":[16148,16148],"name":"31_6","val":"static task"},{"timeRange":[16748,16748],"name":"31_7","val":"static task"},{"timeRange":[16748,16749],"name":"31_8","val":"condition task"},{"timeRange":[16749,16749],"name":"31_9","val":"static task"},{"timeRange":[17122,17123],"name":"31_10","val":"static task"},{"timeRange":[17123,17123],"name":"31_11","val":"condition task"},{"timeRange":[17124,17124],"name":"31_12","val":"static task"},{"timeRange":[17434,17435],"name":"31_13","val":"static task"},{"timeRange":[17435,17435],"name":"31_14","val":"condition task"},{"timeRange":[17435,17436],"name":"31_15","val":"static task"},{"timeRange":[17879,17880],"name":"31_16","val":"static task"},{"timeRange":[17880,17880],"name":"31_17","val":"condition task"},{"timeRange":[17880,17881],"name":"31_18","val":"static task"},{"timeRange":[18196,18197],"name":"31_19","val":"static task"},{"timeRange":[18197,18197],"name":"31_20","val":"condition task"},{"timeRange":[18197,18197],"name":"31_21","val":"static task"},{"timeRange":[18474,18474],"name":"31_22","val":"static task"},{"timeRange":[18474,18475],"name":"31_23","val":"condition task"},{"timeRange":[18475,18475],"name":"31_24","val":"static task"},{"timeRange":[18771,18771],"name":"31_25","val":"static task"},{"timeRange":[18771,18772],"name":"31_26","val":"condition task"},{"timeRange":[19013,19238],"name":"31_27","val":"static task"},{"timeRange":[19238,19239],"name":"31_28","val":"static task"},{"timeRange":[19239,40485],"name":"31_29","val":"static task"},{"timeRange":[40495,40495],"name":"31_30","val":"static task"},{"timeRange":[40496,40496],"name":"31_31","val":"static task"},{"timeRange":[40496,40508],"name":"31_32","val":"static task"},{"timeRange":[40508,40509],"name":"31_33","val":"static task"},{"timeRange":[40509,40509],"name":"31_34","val":"static task"},{"timeRange":[40509,40509],"name":"31_35","val":"static task"},{"timeRange":[40509,40510],"name":"31_36","val":"static task"},{"timeRange":[40510,40510],"name":"31_37","val":"static task"},{"timeRange":[40510,40510],"name":"31_38","val":"static task"},{"timeRange":[40510,40510],"name":"31_39","val":"static task"},{"timeRange":[40511,40511],"name":"31_40","val":"static task"},{"timeRange":[40511,40511],"name":"31_41","val":"static task"},{"timeRange":[40511,40511],"name":"31_42","val":"static task"},{"timeRange":[40511,40512],"name":"31_43","val":"static task"},{"timeRange":[40512,40512],"name":"31_44","val":"static task"},{"timeRange":[40512,40512],"name":"31_45","val":"static task"},{"timeRange":[40512,40512],"name":"31_46","val":"static task"},{"timeRange":[40512,40512],"name":"31_47","val":"static task"},{"timeRange":[40512,40512],"name":"31_48","val":"static task"},{"timeRange":[40512,40513],"name":"31_49","val":"static task"},{"timeRange":[40513,40513],"name":"31_50","val":"static task"},{"timeRange":[40513,40513],"name":"31_51","val":"static task"},{"timeRange":[40513,40514],"name":"31_52","val":"static task"},{"timeRange":[40514,40514],"name":"31_53","val":"static task"},{"timeRange":[40514,40514],"name":"31_54","val":"static task"},{"timeRange":[40514,40514],"name":"31_55","val":"static task"},{"timeRange":[40515,40515],"name":"31_56","val":"static task"},{"timeRange":[40515,40515],"name":"31_57","val":"static task"},{"timeRange":[40515,40515],"name":"31_58","val":"static task"},{"timeRange":[40515,40515],"name":"31_59","val":"static task"},{"timeRange":[40515,40515],"name":"31_60","val":"static task"},{"timeRange":[40515,40515],"name":"31_61","val":"static task"},{"timeRange":[40516,40516],"name":"31_62","val":"static task"},{"timeRange":[40516,40516],"name":"31_63","val":"static task"},{"timeRange":[40516,40524],"name":"31_64","val":"static task"},{"timeRange":[40524,40524],"name":"31_65","val":"static task"},{"timeRange":[40524,40524],"name":"31_66","val":"static task"},{"timeRange":[40524,40524],"name":"31_67","val":"static task"},{"timeRange":[40525,40525],"name":"31_68","val":"static task"},{"timeRange":[40525,40525],"name":"31_69","val":"static task"},{"timeRange":[40525,40525],"name":"31_70","val":"static task"},{"timeRange":[40526,40526],"name":"31_71","val":"static task"},{"timeRange":[40526,40526],"name":"31_72","val":"static task"},{"timeRange":[40526,40526],"name":"31_73","val":"static task"},{"timeRange":[40526,40526],"name":"31_74","val":"static task"},{"timeRange":[40526,40526],"name":"31_75","val":"static task"},{"timeRange":[40527,40527],"name":"31_76","val":"static task"},{"timeRange":[40527,40527],"name":"31_77","val":"static task"},{"timeRange":[40527,40527],"name":"31_78","val":"static task"},{"timeRange":[40527,40527],"name":"31_79","val":"static task"},{"timeRange":[40527,40527],"name":"31_80","val":"static task"},{"timeRange":[40528,40528],"name":"31_81","val":"static task"},{"timeRange":[40528,40528],"name":"31_82","val":"static task"},{"timeRange":[40528,40528],"name":"31_83","val":"static task"},{"timeRange":[40528,40528],"name":"31_84","val":"static task"},{"timeRange":[40529,40529],"name":"31_85","val":"static task"},{"timeRange":[40529,40529],"name":"31_86","val":"static task"},{"timeRange":[40529,40529],"name":"31_87","val":"static task"},{"timeRange":[40529,40529],"name":"31_88","val":"static task"},{"timeRange":[40529,40529],"name":"31_89","val":"static task"},{"timeRange":[40530,40530],"name":"31_90","val":"static task"},{"timeRange":[40530,40530],"name":"31_91","val":"static task"},{"timeRange":[40530,40530],"name":"31_92","val":"static task"},{"timeRange":[40530,40530],"name":"31_93","val":"static task"},{"timeRange":[40530,40530],"name":"31_94","val":"static task"},{"timeRange":[40530,40530],"name":"31_95","val":"static task"},{"timeRange":[40531,40531],"name":"31_96","val":"static task"},{"timeRange":[40531,40531],"name":"31_97","val":"static task"},{"timeRange":[40531,40531],"name":"31_98","val":"static task"},{"timeRange":[40531,40531],"name":"31_99","val":"static task"},{"timeRange":[40532,40532],"name":"31_100","val":"static task"},{"timeRange":[40532,40532],"name":"31_101","val":"static task"},{"timeRange":[40532,40532],"name":"31_102","val":"static task"},{"timeRange":[40532,40532],"name":"31_103","val":"static task"},{"timeRange":[40532,40532],"name":"31_104","val":"static task"},{"timeRange":[40533,40533],"name":"31_105","val":"static task"},{"timeRange":[40533,40533],"name":"31_106","val":"static task"},{"timeRange":[40533,40533],"name":"31_107","val":"static task"},{"timeRange":[40533,40533],"name":"31_108","val":"static task"},{"timeRange":[40533,40533],"name":"31_109","val":"static task"},{"timeRange":[40533,40533],"name":"31_110","val":"static task"},{"timeRange":[40534,40534],"name":"31_111","val":"static task"},{"timeRange":[40534,40534],"name":"31_112","val":"static task"},{"timeRange":[40534,40534],"name":"31_113","val":"static task"},{"timeRange":[40534,40534],"name":"31_114","val":"static task"},{"timeRange":[40535,40535],"name":"31_115","val":"static task"},{"timeRange":[40535,40535],"name":"31_116","val":"static task"},{"timeRange":[40535,40535],"name":"31_117","val":"static task"},{"timeRange":[40535,40535],"name":"31_118","val":"static task"},{"timeRange":[40535,40535],"name":"31_119","val":"static task"},{"timeRange":[40536,40536],"name":"31_120","val":"static task"},{"timeRange":[40536,40536],"name":"31_121","val":"static task"},{"timeRange":[40536,40536],"name":"31_122","val":"static task"},{"timeRange":[40536,40536],"name":"31_123","val":"static task"},{"timeRange":[40536,40536],"name":"31_124","val":"static task"},{"timeRange":[40537,40537],"name":"31_125","val":"static task"},{"timeRange":[40537,40537],"name":"31_126","val":"static task"},{"timeRange":[40537,40537],"name":"31_127","val":"static task"},{"timeRange":[40537,40545],"name":"31_128","val":"static task"},{"timeRange":[40545,40545],"name":"31_129","val":"static task"},{"timeRange":[40545,40545],"name":"31_130","val":"static task"},{"timeRange":[40546,40546],"name":"31_131","val":"static task"},{"timeRange":[40546,40546],"name":"31_132","val":"static task"},{"timeRange":[40546,40546],"name":"31_133","val":"static task"},{"timeRange":[40546,40546],"name":"31_134","val":"static task"},{"timeRange":[40546,40547],"name":"31_135","val":"static task"},{"timeRange":[40547,40547],"name":"31_136","val":"static task"},{"timeRange":[40547,40547],"name":"31_137","val":"static task"},{"timeRange":[40547,40547],"name":"31_138","val":"static task"},{"timeRange":[40547,40547],"name":"31_139","val":"static task"},{"timeRange":[40547,40548],"name":"31_140","val":"static task"},{"timeRange":[40548,40548],"name":"31_141","val":"static task"},{"timeRange":[40548,40548],"name":"31_142","val":"static task"},{"timeRange":[40548,40548],"name":"31_143","val":"static task"},{"timeRange":[40548,40551],"name":"31_144","val":"static task"},{"timeRange":[40551,40551],"name":"31_145","val":"static task"},{"timeRange":[40551,40551],"name":"31_146","val":"static task"},{"timeRange":[40551,40551],"name":"31_147","val":"static task"},{"timeRange":[40551,40552],"name":"31_148","val":"static task"},{"timeRange":[40552,40552],"name":"31_149","val":"static task"},{"timeRange":[40552,40552],"name":"31_150","val":"static task"},{"timeRange":[40552,40552],"name":"31_151","val":"static task"},{"timeRange":[40552,40552],"name":"31_152","val":"static task"},{"timeRange":[40552,40552],"name":"31_153","val":"static task"},{"timeRange":[40552,40553],"name":"31_154","val":"static task"},{"timeRange":[40553,40553],"name":"31_155","val":"static task"},{"timeRange":[40553,40553],"name":"31_156","val":"static task"},{"timeRange":[40553,40553],"name":"31_157","val":"static task"},{"timeRange":[40554,40554],"name":"31_158","val":"static task"},{"timeRange":[40554,40554],"name":"31_159","val":"static task"},{"timeRange":[40554,40554],"name":"31_160","val":"static task"},{"timeRange":[40554,40554],"name":"31_161","val":"static task"},{"timeRange":[40554,40554],"name":"31_162","val":"static task"},{"timeRange":[40554,40554],"name":"31_163","val":"static task"},{"timeRange":[40555,40555],"name":"31_164","val":"static task"},{"timeRange":[40555,40555],"name":"31_165","val":"static task"},{"timeRange":[40555,40558],"name":"31_166","val":"static task"},{"timeRange":[40558,40560],"name":"31_167","val":"static task"},{"timeRange":[40560,40563],"name":"31_168","val":"static task"},{"timeRange":[40564,40566],"name":"31_169","val":"static task"},{"timeRange":[40567,40569],"name":"31_170","val":"static task"},{"timeRange":[40570,40573],"name":"31_171","val":"static task"},{"timeRange":[40574,40579],"name":"31_172","val":"static task"},{"timeRange":[40579,40586],"name":"31_173","val":"static task"},{"timeRange":[40587,40597],"name":"31_174","val":"static task"},{"timeRange":[40598,40607],"name":"31_175","val":"static task"},{"timeRange":[40607,40623],"name":"31_176","val":"static task"},{"timeRange":[40623,40639],"name":"31_177","val":"static task"},{"timeRange":[40687,47086],"name":"31_178","val":"static task"},{"timeRange":[56512,64026],"name":"31_179","val":"static task"}]},{"label":"worker 32","data":[{"timeRange":[40710,55483],"name":"32_0","val":"static task"},{"timeRange":[55485,55487],"name":"32_1","val":"static task"},{"timeRange":[55487,55488],"name":"32_2","val":"static task"},{"timeRange":[55526,55528],"name":"32_3","val":"static task"},{"timeRange":[55529,55530],"name":"32_4","val":"static task"},{"timeRange":[55530,55530],"name":"32_5","val":"static task"},{"timeRange":[55530,55531],"name":"32_6","val":"static task"},{"timeRange":[55532,55532],"name":"32_7","val":"static task"},{"timeRange":[55532,55532],"name":"32_8","val":"static task"},{"timeRange":[55533,55533],"name":"32_9","val":"static task"},{"timeRange":[55534,55534],"name":"32_10","val":"static task"},{"timeRange":[55534,55535],"name":"32_11","val":"static task"},{"timeRange":[55535,55536],"name":"32_12","val":"static task"},{"timeRange":[55536,55536],"name":"32_13","val":"static task"},{"timeRange":[55536,55537],"name":"32_14","val":"static task"},{"timeRange":[55537,55538],"name":"32_15","val":"static task"},{"timeRange":[55538,55538],"name":"32_16","val":"static task"},{"timeRange":[55538,55539],"name":"32_17","val":"static task"},{"timeRange":[55539,55540],"name":"32_18","val":"static task"},{"timeRange":[55540,55540],"name":"32_19","val":"static task"},{"timeRange":[55540,55541],"name":"32_20","val":"static task"},{"timeRange":[55541,55542],"name":"32_21","val":"static task"},{"timeRange":[55542,55542],"name":"32_22","val":"static task"},{"timeRange":[55542,55543],"name":"32_23","val":"static task"},{"timeRange":[55543,55544],"name":"32_24","val":"static task"},{"timeRange":[55544,55544],"name":"32_25","val":"static task"},{"timeRange":[55544,55545],"name":"32_26","val":"static task"},{"timeRange":[55546,55546],"name":"32_27","val":"static task"},{"timeRange":[55546,55547],"name":"32_28","val":"static task"},{"timeRange":[55547,55547],"name":"32_29","val":"static task"},{"timeRange":[55548,55548],"name":"32_30","val":"static task"},{"timeRange":[55548,55548],"name":"32_31","val":"static task"},{"timeRange":[55549,55571],"name":"32_32","val":"static task"},{"timeRange":[55571,55572],"name":"32_33","val":"static task"},{"timeRange":[55572,55572],"name":"32_34","val":"static task"},{"timeRange":[55572,55573],"name":"32_35","val":"static task"},{"timeRange":[55573,55573],"name":"32_36","val":"static task"},{"timeRange":[55574,55574],"name":"32_37","val":"static task"},{"timeRange":[55574,55575],"name":"32_38","val":"static task"},{"timeRange":[55575,55576],"name":"32_39","val":"static task"},{"timeRange":[55576,55577],"name":"32_40","val":"static task"},{"timeRange":[55577,55578],"name":"32_41","val":"static task"},{"timeRange":[55578,55578],"name":"32_42","val":"static task"},{"timeRange":[55579,55579],"name":"32_43","val":"static task"},{"timeRange":[55579,55580],"name":"32_44","val":"static task"},{"timeRange":[55580,55580],"name":"32_45","val":"static task"},{"timeRange":[55581,55581],"name":"32_46","val":"static task"},{"timeRange":[55582,55582],"name":"32_47","val":"static task"},{"timeRange":[55582,55583],"name":"32_48","val":"static task"},{"timeRange":[55583,55584],"name":"32_49","val":"static task"},{"timeRange":[55584,55584],"name":"32_50","val":"static task"},{"timeRange":[55584,55585],"name":"32_51","val":"static task"},{"timeRange":[55585,55586],"name":"32_52","val":"static task"},{"timeRange":[55586,55586],"name":"32_53","val":"static task"},{"timeRange":[55586,55587],"name":"32_54","val":"static task"},{"timeRange":[55587,55588],"name":"32_55","val":"static task"},{"timeRange":[55588,55588],"name":"32_56","val":"static task"},{"timeRange":[55589,55589],"name":"32_57","val":"static task"},{"timeRange":[55589,55590],"name":"32_58","val":"static task"},{"timeRange":[55590,55590],"name":"32_59","val":"static task"},{"timeRange":[55590,55591],"name":"32_60","val":"static task"},{"timeRange":[55591,55592],"name":"32_61","val":"static task"},{"timeRange":[55592,55592],"name":"32_62","val":"static task"},{"timeRange":[55592,55593],"name":"32_63","val":"static task"},{"timeRange":[55593,55612],"name":"32_64","val":"static task"},{"timeRange":[55612,55613],"name":"32_65","val":"static task"},{"timeRange":[55613,55613],"name":"32_66","val":"static task"},{"timeRange":[55614,55614],"name":"32_67","val":"static task"},{"timeRange":[55615,55615],"name":"32_68","val":"static task"},{"timeRange":[55615,55616],"name":"32_69","val":"static task"},{"timeRange":[55616,55617],"name":"32_70","val":"static task"},{"timeRange":[55617,55617],"name":"32_71","val":"static task"},{"timeRange":[55617,55618],"name":"32_72","val":"static task"},{"timeRange":[55618,55619],"name":"32_73","val":"static task"},{"timeRange":[55619,55619],"name":"32_74","val":"static task"},{"timeRange":[55619,55620],"name":"32_75","val":"static task"},{"timeRange":[55620,55621],"name":"32_76","val":"static task"},{"timeRange":[55621,55621],"name":"32_77","val":"static task"},{"timeRange":[55621,55622],"name":"32_78","val":"static task"},{"timeRange":[55622,55623],"name":"32_79","val":"static task"},{"timeRange":[55623,55623],"name":"32_80","val":"static task"},{"timeRange":[55624,55624],"name":"32_81","val":"static task"},{"timeRange":[55625,55625],"name":"32_82","val":"static task"},{"timeRange":[55626,55627],"name":"32_83","val":"static task"},{"timeRange":[55627,55627],"name":"32_84","val":"static task"},{"timeRange":[55627,55628],"name":"32_85","val":"static task"},{"timeRange":[55628,55628],"name":"32_86","val":"static task"},{"timeRange":[55629,55629],"name":"32_87","val":"static task"},{"timeRange":[55629,55629],"name":"32_88","val":"static task"},{"timeRange":[55630,55630],"name":"32_89","val":"static task"},{"timeRange":[55631,55631],"name":"32_90","val":"static task"},{"timeRange":[55631,55631],"name":"32_91","val":"static task"},{"timeRange":[55632,55632],"name":"32_92","val":"static task"},{"timeRange":[55632,55633],"name":"32_93","val":"static task"},{"timeRange":[55633,55633],"name":"32_94","val":"static task"},{"timeRange":[55633,55634],"name":"32_95","val":"static task"},{"timeRange":[55634,55634],"name":"32_96","val":"static task"},{"timeRange":[55634,55635],"name":"32_97","val":"static task"},{"timeRange":[55635,55635],"name":"32_98","val":"static task"},{"timeRange":[55635,55636],"name":"32_99","val":"static task"},{"timeRange":[55636,55636],"name":"32_100","val":"static task"},{"timeRange":[55636,55637],"name":"32_101","val":"static task"},{"timeRange":[55637,55637],"name":"32_102","val":"static task"},{"timeRange":[55637,55638],"name":"32_103","val":"static task"},{"timeRange":[55638,55638],"name":"32_104","val":"static task"},{"timeRange":[55638,55639],"name":"32_105","val":"static task"},{"timeRange":[55639,55639],"name":"32_106","val":"static task"},{"timeRange":[55639,55639],"name":"32_107","val":"static task"},{"timeRange":[55640,55640],"name":"32_108","val":"static task"},{"timeRange":[55640,55640],"name":"32_109","val":"static task"},{"timeRange":[55641,55642],"name":"32_110","val":"static task"},{"timeRange":[55642,55642],"name":"32_111","val":"static task"},{"timeRange":[55642,55643],"name":"32_112","val":"static task"},{"timeRange":[55643,55643],"name":"32_113","val":"static task"},{"timeRange":[55644,55644],"name":"32_114","val":"static task"},{"timeRange":[55644,55645],"name":"32_115","val":"static task"},{"timeRange":[55645,55645],"name":"32_116","val":"static task"},{"timeRange":[55645,55646],"name":"32_117","val":"static task"},{"timeRange":[55646,55646],"name":"32_118","val":"static task"},{"timeRange":[55646,55647],"name":"32_119","val":"static task"},{"timeRange":[55647,55647],"name":"32_120","val":"static task"},{"timeRange":[55648,55648],"name":"32_121","val":"static task"},{"timeRange":[55648,55648],"name":"32_122","val":"static task"},{"timeRange":[55649,55649],"name":"32_123","val":"static task"},{"timeRange":[55650,55650],"name":"32_124","val":"static task"},{"timeRange":[55650,55651],"name":"32_125","val":"static task"},{"timeRange":[55651,55651],"name":"32_126","val":"static task"},{"timeRange":[55652,55652],"name":"32_127","val":"static task"},{"timeRange":[55652,55680],"name":"32_128","val":"static task"},{"timeRange":[55680,55681],"name":"32_129","val":"static task"},{"timeRange":[55681,55681],"name":"32_130","val":"static task"},{"timeRange":[55681,55682],"name":"32_131","val":"static task"},{"timeRange":[55682,55682],"name":"32_132","val":"static task"},{"timeRange":[55682,55683],"name":"32_133","val":"static task"},{"timeRange":[55683,55684],"name":"32_134","val":"static task"},{"timeRange":[55684,55684],"name":"32_135","val":"static task"},{"timeRange":[55684,55685],"name":"32_136","val":"static task"},{"timeRange":[55685,55685],"name":"32_137","val":"static task"},{"timeRange":[55685,55686],"name":"32_138","val":"static task"},{"timeRange":[55686,55727],"name":"32_139","val":"static task"},{"timeRange":[55728,55784],"name":"32_140","val":"static task"},{"timeRange":[55784,55990],"name":"32_141","val":"static task"},{"timeRange":[55990,56457],"name":"32_142","val":"static task"},{"timeRange":[56457,56736],"name":"32_143","val":"static task"}]},{"label":"worker 33","data":[{"timeRange":[154,3857],"name":"33_0","val":"static task"},{"timeRange":[40529,40540],"name":"33_1","val":"static task"},{"timeRange":[40548,40561],"name":"33_2","val":"static task"},{"timeRange":[40566,40583],"name":"33_3","val":"static task"},{"timeRange":[40588,40605],"name":"33_4","val":"static task"},{"timeRange":[40613,40629],"name":"33_5","val":"static task"},{"timeRange":[40707,55011],"name":"33_6","val":"static task"},{"timeRange":[55642,65811],"name":"33_7","val":"static task"},{"timeRange":[89713,89940],"name":"33_8","val":"static task"}]},{"label":"worker 34","data":[{"timeRange":[171,2898],"name":"34_0","val":"static task"},{"timeRange":[40612,40636],"name":"34_1","val":"static task"},{"timeRange":[40760,54187],"name":"34_2","val":"static task"},{"timeRange":[55664,70980],"name":"34_3","val":"static task"},{"timeRange":[89437,89599],"name":"34_4","val":"static task"},{"timeRange":[89604,89879],"name":"34_5","val":"static task"}]},{"label":"worker 35","data":[{"timeRange":[148,1686],"name":"35_0","val":"static task"},{"timeRange":[40631,40664],"name":"35_1","val":"static task"},{"timeRange":[40666,40666],"name":"35_2","val":"static task"},{"timeRange":[40666,40667],"name":"35_3","val":"static task"},{"timeRange":[40684,40685],"name":"35_4","val":"static task"},{"timeRange":[40685,40685],"name":"35_5","val":"static task"},{"timeRange":[40686,40686],"name":"35_6","val":"static task"},{"timeRange":[40686,40686],"name":"35_7","val":"static task"},{"timeRange":[40687,40687],"name":"35_8","val":"static task"},{"timeRange":[40687,40687],"name":"35_9","val":"static task"},{"timeRange":[40688,40688],"name":"35_10","val":"static task"},{"timeRange":[40688,40689],"name":"35_11","val":"static task"},{"timeRange":[40689,40689],"name":"35_12","val":"static task"},{"timeRange":[40689,40690],"name":"35_13","val":"static task"},{"timeRange":[40690,40690],"name":"35_14","val":"static task"},{"timeRange":[40690,40691],"name":"35_15","val":"static task"},{"timeRange":[40691,40691],"name":"35_16","val":"static task"},{"timeRange":[40691,40691],"name":"35_17","val":"static task"},{"timeRange":[40691,40692],"name":"35_18","val":"static task"},{"timeRange":[40692,40692],"name":"35_19","val":"static task"},{"timeRange":[40693,40693],"name":"35_20","val":"static task"},{"timeRange":[40693,40693],"name":"35_21","val":"static task"},{"timeRange":[40694,40694],"name":"35_22","val":"static task"},{"timeRange":[40694,40694],"name":"35_23","val":"static task"},{"timeRange":[40694,40694],"name":"35_24","val":"static task"},{"timeRange":[40695,40695],"name":"35_25","val":"static task"},{"timeRange":[40695,40695],"name":"35_26","val":"static task"},{"timeRange":[40696,40696],"name":"35_27","val":"static task"},{"timeRange":[40696,40696],"name":"35_28","val":"static task"},{"timeRange":[40697,40697],"name":"35_29","val":"static task"},{"timeRange":[40697,40697],"name":"35_30","val":"static task"},{"timeRange":[40698,40698],"name":"35_31","val":"static task"},{"timeRange":[40698,40714],"name":"35_32","val":"static task"},{"timeRange":[40714,40714],"name":"35_33","val":"static task"},{"timeRange":[40715,40715],"name":"35_34","val":"static task"},{"timeRange":[40715,40715],"name":"35_35","val":"static task"},{"timeRange":[40716,40716],"name":"35_36","val":"static task"},{"timeRange":[40716,40716],"name":"35_37","val":"static task"},{"timeRange":[40716,40717],"name":"35_38","val":"static task"},{"timeRange":[40717,40717],"name":"35_39","val":"static task"},{"timeRange":[40717,40717],"name":"35_40","val":"static task"},{"timeRange":[40717,40718],"name":"35_41","val":"static task"},{"timeRange":[40718,40718],"name":"35_42","val":"static task"},{"timeRange":[40718,40718],"name":"35_43","val":"static task"},{"timeRange":[40719,40719],"name":"35_44","val":"static task"},{"timeRange":[40719,40719],"name":"35_45","val":"static task"},{"timeRange":[40720,40720],"name":"35_46","val":"static task"},{"timeRange":[40720,40720],"name":"35_47","val":"static task"},{"timeRange":[40720,40721],"name":"35_48","val":"static task"},{"timeRange":[40721,40721],"name":"35_49","val":"static task"},{"timeRange":[40721,40721],"name":"35_50","val":"static task"},{"timeRange":[40721,40722],"name":"35_51","val":"static task"},{"timeRange":[40722,40722],"name":"35_52","val":"static task"},{"timeRange":[40722,40722],"name":"35_53","val":"static task"},{"timeRange":[40722,40723],"name":"35_54","val":"static task"},{"timeRange":[40723,40723],"name":"35_55","val":"static task"},{"timeRange":[40723,40723],"name":"35_56","val":"static task"},{"timeRange":[40723,40724],"name":"35_57","val":"static task"},{"timeRange":[40724,40724],"name":"35_58","val":"static task"},{"timeRange":[40724,40724],"name":"35_59","val":"static task"},{"timeRange":[40725,40725],"name":"35_60","val":"static task"},{"timeRange":[40725,40725],"name":"35_61","val":"static task"},{"timeRange":[40725,40726],"name":"35_62","val":"static task"},{"timeRange":[40726,40726],"name":"35_63","val":"static task"},{"timeRange":[40726,40850],"name":"35_64","val":"static task"},{"timeRange":[40851,40851],"name":"35_65","val":"static task"},{"timeRange":[40851,40851],"name":"35_66","val":"static task"},{"timeRange":[40852,40852],"name":"35_67","val":"static task"},{"timeRange":[40852,40852],"name":"35_68","val":"static task"},{"timeRange":[40852,40853],"name":"35_69","val":"static task"},{"timeRange":[40853,40853],"name":"35_70","val":"static task"},{"timeRange":[40853,40853],"name":"35_71","val":"static task"},{"timeRange":[40853,40854],"name":"35_72","val":"static task"},{"timeRange":[40854,40854],"name":"35_73","val":"static task"},{"timeRange":[40854,40854],"name":"35_74","val":"static task"},{"timeRange":[40855,40855],"name":"35_75","val":"static task"},{"timeRange":[40855,40855],"name":"35_76","val":"static task"},{"timeRange":[40855,40856],"name":"35_77","val":"static task"},{"timeRange":[40856,40856],"name":"35_78","val":"static task"},{"timeRange":[40856,40856],"name":"35_79","val":"static task"},{"timeRange":[40857,40857],"name":"35_80","val":"static task"},{"timeRange":[40857,40857],"name":"35_81","val":"static task"},{"timeRange":[40857,40857],"name":"35_82","val":"static task"},{"timeRange":[40858,40858],"name":"35_83","val":"static task"},{"timeRange":[40858,40858],"name":"35_84","val":"static task"},{"timeRange":[40858,40858],"name":"35_85","val":"static task"},{"timeRange":[40859,40859],"name":"35_86","val":"static task"},{"timeRange":[40859,40859],"name":"35_87","val":"static task"},{"timeRange":[40859,40859],"name":"35_88","val":"static task"},{"timeRange":[40860,40860],"name":"35_89","val":"static task"},{"timeRange":[40860,40860],"name":"35_90","val":"static task"},{"timeRange":[40860,40860],"name":"35_91","val":"static task"},{"timeRange":[40861,40861],"name":"35_92","val":"static task"},{"timeRange":[40861,40861],"name":"35_93","val":"static task"},{"timeRange":[40862,40862],"name":"35_94","val":"static task"},{"timeRange":[40862,40862],"name":"35_95","val":"static task"},{"timeRange":[40862,40862],"name":"35_96","val":"static task"},{"timeRange":[40863,40863],"name":"35_97","val":"static task"},{"timeRange":[40863,40863],"name":"35_98","val":"static task"},{"timeRange":[40863,40864],"name":"35_99","val":"static task"},{"timeRange":[40864,40864],"name":"35_100","val":"static task"},{"timeRange":[40864,40864],"name":"35_101","val":"static task"},{"timeRange":[40864,40865],"name":"35_102","val":"static task"},{"timeRange":[40865,40865],"name":"35_103","val":"static task"},{"timeRange":[40865,40865],"name":"35_104","val":"static task"},{"timeRange":[40865,40866],"name":"35_105","val":"static task"},{"timeRange":[40866,40866],"name":"35_106","val":"static task"},{"timeRange":[40866,40866],"name":"35_107","val":"static task"},{"timeRange":[40867,40867],"name":"35_108","val":"static task"},{"timeRange":[40867,40867],"name":"35_109","val":"static task"},{"timeRange":[40867,40868],"name":"35_110","val":"static task"},{"timeRange":[40868,40868],"name":"35_111","val":"static task"},{"timeRange":[40868,40868],"name":"35_112","val":"static task"},{"timeRange":[40868,40869],"name":"35_113","val":"static task"},{"timeRange":[40869,40869],"name":"35_114","val":"static task"},{"timeRange":[40869,40869],"name":"35_115","val":"static task"},{"timeRange":[40869,40870],"name":"35_116","val":"static task"},{"timeRange":[40870,40870],"name":"35_117","val":"static task"},{"timeRange":[40870,40870],"name":"35_118","val":"static task"},{"timeRange":[40870,40871],"name":"35_119","val":"static task"},{"timeRange":[40871,40871],"name":"35_120","val":"static task"},{"timeRange":[40871,40871],"name":"35_121","val":"static task"},{"timeRange":[40872,40872],"name":"35_122","val":"static task"},{"timeRange":[40872,40872],"name":"35_123","val":"static task"},{"timeRange":[40872,40872],"name":"35_124","val":"static task"},{"timeRange":[40873,40873],"name":"35_125","val":"static task"},{"timeRange":[40873,40873],"name":"35_126","val":"static task"},{"timeRange":[40874,40874],"name":"35_127","val":"static task"},{"timeRange":[40874,40922],"name":"35_128","val":"static task"},{"timeRange":[40922,40922],"name":"35_129","val":"static task"},{"timeRange":[40923,40923],"name":"35_130","val":"static task"},{"timeRange":[40923,40923],"name":"35_131","val":"static task"},{"timeRange":[40923,40924],"name":"35_132","val":"static task"},{"timeRange":[40924,40924],"name":"35_133","val":"static task"},{"timeRange":[40924,40925],"name":"35_134","val":"static task"},{"timeRange":[40925,40925],"name":"35_135","val":"static task"},{"timeRange":[40925,40926],"name":"35_136","val":"static task"},{"timeRange":[40926,40926],"name":"35_137","val":"static task"},{"timeRange":[40926,40958],"name":"35_138","val":"static task"},{"timeRange":[40959,40959],"name":"35_139","val":"static task"},{"timeRange":[40959,41040],"name":"35_140","val":"static task"},{"timeRange":[41041,41152],"name":"35_141","val":"static task"},{"timeRange":[41153,41370],"name":"35_142","val":"static task"}]},{"label":"worker 36","data":[{"timeRange":[144,1190],"name":"36_0","val":"static task"},{"timeRange":[40689,53690],"name":"36_1","val":"static task"},{"timeRange":[55777,69131],"name":"36_2","val":"static task"},{"timeRange":[89597,89862],"name":"36_3","val":"static task"}]},{"label":"worker 37","data":[{"timeRange":[143,3197],"name":"37_0","val":"static task"},{"timeRange":[40632,40651],"name":"37_1","val":"static task"},{"timeRange":[40727,47866],"name":"37_2","val":"static task"},{"timeRange":[56411,65467],"name":"37_3","val":"static task"}]},{"label":"worker 38","data":[{"timeRange":[161,2228],"name":"38_0","val":"static task"},{"timeRange":[40726,52033],"name":"38_1","val":"static task"},{"timeRange":[55851,69503],"name":"38_2","val":"static task"},{"timeRange":[89548,89710],"name":"38_3","val":"static task"},{"timeRange":[89711,89974],"name":"38_4","val":"static task"},{"timeRange":[97875,98246],"name":"38_5","val":"static task"}]},{"label":"worker 39","data":[{"timeRange":[135,1133],"name":"39_0","val":"static task"},{"timeRange":[40696,53479],"name":"39_1","val":"static task"},{"timeRange":[55751,84324],"name":"39_2","val":"static task"},{"timeRange":[89358,89515],"name":"39_3","val":"static task"},{"timeRange":[89521,89766],"name":"39_4","val":"static task"}]},{"label":"worker 40","data":[{"timeRange":[4133,9679],"name":"40_0","val":"cudaflow"},{"timeRange":[12914,13464],"name":"40_1","val":"cudaflow"},{"timeRange":[13650,14866],"name":"40_2","val":"cudaflow"},{"timeRange":[15060,15603],"name":"40_3","val":"cudaflow"},{"timeRange":[15665,16120],"name":"40_4","val":"cudaflow"},{"timeRange":[16310,16722],"name":"40_5","val":"cudaflow"},{"timeRange":[16764,17099],"name":"40_6","val":"cudaflow"},{"timeRange":[17129,17412],"name":"40_7","val":"cudaflow"},{"timeRange":[17459,17741],"name":"40_8","val":"cudaflow"},{"timeRange":[17930,18172],"name":"40_9","val":"cudaflow"},{"timeRange":[18231,18449],"name":"40_10","val":"cudaflow"},{"timeRange":[18528,18746],"name":"40_11","val":"cudaflow"},{"timeRange":[18883,18993],"name":"40_12","val":"cudaflow"}]}]}];

  const matmul=[{"group":"executor 0","data":[{"label":"worker 0","data":[{"timeRange":[78,103],"name":"0_0","val":"static task"},{"timeRange":[103,129],"name":"0_1","val":"static task"},{"timeRange":[129,154],"name":"0_2","val":"static task"},{"timeRange":[154,179],"name":"0_3","val":"static task"},{"timeRange":[180,205],"name":"0_4","val":"static task"},{"timeRange":[205,230],"name":"0_5","val":"static task"},{"timeRange":[230,256],"name":"0_6","val":"static task"},{"timeRange":[256,281],"name":"0_7","val":"static task"}]},{"label":"worker 1","data":[{"timeRange":[80,106],"name":"1_0","val":"static task"},{"timeRange":[106,131],"name":"1_1","val":"static task"},{"timeRange":[132,157],"name":"1_2","val":"static task"},{"timeRange":[157,182],"name":"1_3","val":"static task"},{"timeRange":[183,208],"name":"1_4","val":"static task"},{"timeRange":[208,233],"name":"1_5","val":"static task"},{"timeRange":[234,259],"name":"1_6","val":"static task"},{"timeRange":[259,286],"name":"1_7","val":"static task"}]},{"label":"worker 2","data":[{"timeRange":[76,101],"name":"2_0","val":"static task"},{"timeRange":[102,127],"name":"2_1","val":"static task"},{"timeRange":[127,152],"name":"2_2","val":"static task"},{"timeRange":[153,178],"name":"2_3","val":"static task"},{"timeRange":[178,203],"name":"2_4","val":"static task"},{"timeRange":[203,229],"name":"2_5","val":"static task"},{"timeRange":[229,254],"name":"2_6","val":"static task"},{"timeRange":[254,279],"name":"2_7","val":"static task"}]},{"label":"worker 3","data":[{"timeRange":[76,102],"name":"3_0","val":"static task"},{"timeRange":[102,127],"name":"3_1","val":"static task"},{"timeRange":[127,153],"name":"3_2","val":"static task"},{"timeRange":[153,178],"name":"3_3","val":"static task"},{"timeRange":[179,204],"name":"3_4","val":"static task"},{"timeRange":[204,230],"name":"3_5","val":"static task"},{"timeRange":[230,255],"name":"3_6","val":"static task"},{"timeRange":[255,282],"name":"3_7","val":"static task"}]},{"label":"worker 4","data":[{"timeRange":[74,100],"name":"4_0","val":"static task"},{"timeRange":[101,126],"name":"4_1","val":"static task"},{"timeRange":[126,151],"name":"4_2","val":"static task"},{"timeRange":[152,177],"name":"4_3","val":"static task"},{"timeRange":[177,202],"name":"4_4","val":"static task"},{"timeRange":[203,228],"name":"4_5","val":"static task"},{"timeRange":[228,254],"name":"4_6","val":"static task"},{"timeRange":[254,279],"name":"4_7","val":"static task"}]},{"label":"worker 5","data":[{"timeRange":[70,92],"name":"5_0","val":"static task"},{"timeRange":[92,118],"name":"5_1","val":"static task"},{"timeRange":[118,143],"name":"5_2","val":"static task"},{"timeRange":[143,168],"name":"5_3","val":"static task"},{"timeRange":[169,194],"name":"5_4","val":"static task"},{"timeRange":[194,219],"name":"5_5","val":"static task"},{"timeRange":[220,245],"name":"5_6","val":"static task"},{"timeRange":[245,270],"name":"5_7","val":"static task"}]},{"label":"worker 6","data":[{"timeRange":[67,89],"name":"6_0","val":"static task"},{"timeRange":[89,115],"name":"6_1","val":"static task"},{"timeRange":[115,140],"name":"6_2","val":"static task"},{"timeRange":[140,166],"name":"6_3","val":"static task"},{"timeRange":[166,191],"name":"6_4","val":"static task"},{"timeRange":[191,217],"name":"6_5","val":"static task"},{"timeRange":[217,242],"name":"6_6","val":"static task"},{"timeRange":[242,268],"name":"6_7","val":"static task"}]},{"label":"worker 7","data":[{"timeRange":[24,27],"name":"allocate_a","val":"static task"},{"timeRange":[27,43],"name":"allocate_c","val":"static task"},{"timeRange":[63,89],"name":"7_2","val":"static task"},{"timeRange":[89,114],"name":"7_3","val":"static task"},{"timeRange":[114,139],"name":"7_4","val":"static task"},{"timeRange":[139,165],"name":"7_5","val":"static task"},{"timeRange":[165,190],"name":"7_6","val":"static task"},{"timeRange":[190,215],"name":"7_7","val":"static task"},{"timeRange":[215,240],"name":"7_8","val":"static task"},{"timeRange":[241,266],"name":"7_9","val":"static task"}]},{"label":"worker 8","data":[{"timeRange":[53,68],"name":"8_0","val":"static task"},{"timeRange":[72,98],"name":"8_1","val":"static task"},{"timeRange":[98,123],"name":"8_2","val":"static task"},{"timeRange":[124,149],"name":"8_3","val":"static task"},{"timeRange":[149,174],"name":"8_4","val":"static task"},{"timeRange":[175,200],"name":"8_5","val":"static task"},{"timeRange":[200,225],"name":"8_6","val":"static task"},{"timeRange":[226,251],"name":"8_7","val":"static task"},{"timeRange":[251,276],"name":"8_8","val":"static task"}]},{"label":"worker 9","data":[{"timeRange":[25,49],"name":"allocate_b","val":"static task"},{"timeRange":[52,77],"name":"9_1","val":"static task"},{"timeRange":[77,102],"name":"9_2","val":"static task"},{"timeRange":[103,128],"name":"9_3","val":"static task"},{"timeRange":[128,153],"name":"9_4","val":"static task"},{"timeRange":[153,179],"name":"9_5","val":"static task"},{"timeRange":[179,204],"name":"9_6","val":"static task"},{"timeRange":[204,230],"name":"9_7","val":"static task"},{"timeRange":[230,255],"name":"9_8","val":"static task"},{"timeRange":[255,280],"name":"9_9","val":"static task"}]},{"label":"worker 10","data":[{"timeRange":[51,77],"name":"10_0","val":"static task"},{"timeRange":[77,102],"name":"10_1","val":"static task"},{"timeRange":[103,128],"name":"10_2","val":"static task"},{"timeRange":[128,153],"name":"10_3","val":"static task"},{"timeRange":[154,179],"name":"10_4","val":"static task"},{"timeRange":[179,204],"name":"10_5","val":"static task"},{"timeRange":[205,230],"name":"10_6","val":"static task"},{"timeRange":[230,255],"name":"10_7","val":"static task"},{"timeRange":[256,281],"name":"10_8","val":"static task"}]},{"label":"worker 11","data":[{"timeRange":[59,85],"name":"11_0","val":"static task"},{"timeRange":[86,111],"name":"11_1","val":"static task"},{"timeRange":[111,136],"name":"11_2","val":"static task"},{"timeRange":[137,162],"name":"11_3","val":"static task"},{"timeRange":[162,187],"name":"11_4","val":"static task"},{"timeRange":[188,213],"name":"11_5","val":"static task"},{"timeRange":[213,238],"name":"11_6","val":"static task"},{"timeRange":[239,264],"name":"11_7","val":"static task"},{"timeRange":[264,293],"name":"11_8","val":"static task"}]}]}];

  var state = {
    
    // DOMAIN (data) -> RANGE (graph)

    // main timeline svg
    svg : null,                     // svg block
    graph: null,                    // graph block
    graphW: null,
    graphH: null,
    zoomX: [null, null],            // scoped time data
    zoomY: [null, null],            // scoped line data
    
    // overview element
    overviewAreaSvg: null,
    overviewAreaScale: d3.scaleLinear(),
    overviewAreaSelection: [null, null],
    overviewAreaDomain: [null, null],
    overviewAreaBrush: null,
    overviewAreaTopMargin: 1,
    overviewAreaBottomMargin: 30,
    overviewAreaXGrid: d3.axisBottom().tickFormat(''),
    overviewAreaXAxis: d3.axisBottom().tickPadding(0),
    overviewAreaBrush: d3.brushX(),

    // axes attributes
    xScale: d3.scaleLinear(),
    yScale: d3.scalePoint(),  
    grpScale: d3.scaleOrdinal(),
    xAxis: d3.axisBottom(),
    xGrid: d3.axisTop(),
    yAxis: d3.axisRight(),
    grpAxis: d3.axisLeft(),
    //xTickFormat: n => +n,
    minLabelFont: 2,
    

    // legend
    //zColorMap: null,
    zColorMap: new Map([
      ['static task', '#4682b4'],
      ['dynamic task', '#ff7f0e'],
      ['cudaflow', '#6A0DAD'],
      ['condition task', '#41A317'],
      ['module task', '#0000FF']
    ]),
    zScale: null,
    zGroup: null,
    zWidth: null,
    zHeight: null,
    
    // window attributes
    width: window.innerWidth,
    height: 640,
    maxHeight: Infinity,
    maxLineHeight: 20,
    leftMargin: 100,
    rightMargin: 100,
    topMargin: 26,
    bottomMargin: 30,

    // date marker line
    dateMarker: null,
    dateMarkerLine: null,
    
    // segmenet  
    minSegmentDuration: 0, // ms
    disableHover: false,
    minX: null,
    maxY: null,
    
    // transition
    transDuration: 700,

    // data field
    completeStructData: [],       // groups and lines
    completeFlatData: [],         // flat segments with gropu and line
    structData: null,
    flatData: null,
    totalNLines: 0,
    nLines: 0
  };

  // Procedure: make_timeline_structure
  function make_timeline_structure(dom_element) {
    
    //console.log("timeline chart created at", dom_element);
    
    const elem = d3.select(dom_element)
                   .attr('class', 'timelines-chart');

    // main svg
    state.svg = elem.append('svg');
    
    // overview svg
    state.overviewAreaSvg = elem.append('div').append('svg')
                              .attr('class', 'brusher');
    
    //console.log("_make_timeline_structure");
      
    state.yScale.invert = _invertOrdinal;
    state.grpScale.invert = _invertOrdinal;
    
    // set up the gradient field 
    _make_timeline_gradient_field();

    // set up the axes
    _make_timeline_axes();

    // set up the legend
    _make_timeline_legend();
    
    // set up the main graph area group
    _make_timeline_graph();

    // this seems redundant
    _make_timeline_date_marker_line();
    
    // set up the overview area
    _make_timeline_overview();

    // set up the tooltips
    _make_timeline_tooltips();
    
    // set up the dom events
    _make_timeline_events();
  }

  // Procedure: feed()
  function feed(rawData) {

    // clear the previous state
    state.zoomX = [null, null];
    state.zoomY = [null, null];
    state.minX  = null;
    state.maxX  = null;
    state.completeStructData = [];
    state.completeFlatData = [];
    state.totalNLines = 0;

    for (let i=0, ilen=rawData.length; i<ilen; i++) {
      const group = rawData[i].group;

      state.completeStructData.push({
        group: group,
        lines: rawData[i].data.map(d => d.label)
      });

      for (let j= 0, jlen=rawData[i].data.length; j<jlen; j++) {  // iterate lines
        for (let k= 0, klen=rawData[i].data[j].data.length; k<klen; k++) {  // iterate segs
          const { timeRange, val, name } = rawData[i].data[j].data[k];

          state.completeFlatData.push({
            group: group,
            label: rawData[i].data[j].label,
            timeRange: timeRange,
            val: val,                             // legend value
            //data: rawData[i].data[j].data[k],
            name: name
          });

          if(state.minX == null || timeRange[0] < state.minX) {
            state.minX = timeRange[0];
          }

          if(state.maxX == null || timeRange[1] > state.maxX) {
            state.maxX = timeRange[1];
          }
        }
        state.totalNLines++;
      }
    }

    //console.log("total", state.totalNLines, " lines");
          
    state.overviewAreaDomain = [state.minX, state.maxX];
    update([state.minX, state.maxX], [null, null]);
  }

  // Procedure: update
  function update(zoomX, zoomY) {
    
    // if the successive change is small, we don't update;
    // this also avoids potential infinite loops caused by cyclic event updates
    if((state.zoomX[0] == zoomX[0] && state.zoomX[1] == zoomX[1] &&
        state.zoomY[0] == zoomY[0] && state.zoomY[1] == zoomY[1]) ||
      (Math.abs(state.zoomX[0] - zoomX[0]) < Number.EPSILON && 
       Math.abs(state.zoomX[1] - zoomX[1]) < Number.EPSILON &&
       Math.abs(state.zoomY[0] - zoomY[0]) < Number.EPSILON && 
       Math.abs(state.zoomY[1] - zoomY[1]) < Number.EPSILON)) {
      //console.log("skip update", state.zoomX, state.zoomY, zoomX, zoomY);
      return;
    }
    
    // we use zoomX and zoomY to control the update
    state.zoomX = zoomX;
    state.zoomY = zoomY;
    state.overviewAreaSelection = state.zoomX;

    //console.log("update");

    _apply_filters();
    _adjust_dimensions();
    _adjust_xscale();
    _adjust_yscale();
    _adjust_grpscale();
    _adjust_legend();
      
    _render_axes();
    _render_groups();
    _render_timelines();
    _render_overview_area();
  }

  // ----------------------------------------------------------------------------
  // private function definitions
  // ----------------------------------------------------------------------------

  // Procedure: _invertOrdinal 
  // perform interpolation
  function _invertOrdinal(val, cmpFunc) {

    cmpFunc = cmpFunc || function (a, b) {
        return (a >= b);
      };

    const scDomain = this.domain();
    let scRange = this.range();

    if (scRange.length === 2 && scDomain.length !== 2) {
      // Special case, interpolate range vals
      scRange = d3.range(scRange[0], scRange[1], (scRange[1] - scRange[0]) / scDomain.length);
    }

    const bias = scRange[0];
    for (let i = 0, len = scRange.length; i < len; i++) {
      if (cmpFunc(scRange[i] + bias, val)) {
        return scDomain[Math.round(i * scDomain.length / scRange.length)];
      }
    }

    return this.domain()[this.domain().length-1];
  }
    
  function _make_timeline_gradient_field() {  
    //console.log("making gradient ...");
    state.groupGradId = `areaGradient${Math.round(Math.random()*10000)}`;
    const gradient = state.svg.append('linearGradient');

    gradient.attr('y1', '0%')
            .attr('y2', '100%')
            .attr('x1', '0%')
            .attr('x2', '0%')
            .attr('id', state.groupGradId);
    
    const color_scale = d3.scaleLinear().domain([0, 1]).range(['#FAFAFA', '#E0E0E0']);
    const stop_scale = d3.scaleLinear().domain([0, 100]).range(color_scale.domain());
    
    let color_stops = gradient.selectAll('stop')
                                 .data(d3.range(0, 100.01, 20)); 

    color_stops.exit().remove();
    color_stops.merge(color_stops.enter().append('stop'))
      .attr('offset', d => `${d}%`)
      .attr('stop-color', d => color_scale(stop_scale(d)));
  }

  // Procedure: _make_timeline_date_marker_line
  function _make_timeline_date_marker_line() {
    //console.log("making date marker ...");
    state.dateMarkerLine = state.svg.append('line').attr('class', 'x-axis-date-marker');
  }

  // Procedure: _make_timeline_overview
  function _make_timeline_overview() {
    //console.log("making the overview ...");

    state.overviewAreaBrush
      .handleSize(24)
      .on('end', function() {
        
        //console.log("ON 'end': brush ends by source", d3.event.sourceEvent);

        if (!d3.event.sourceEvent) {
          return;
        }

        //console.log("    -> type:", d3.event.sourceEvent.type);

        const selection = d3.event.selection ? 
          d3.event.selection.map(state.overviewAreaScale.invert) : 
          state.overviewAreaScale.domain();

        // avoid infinite event loop
        if(d3.event.sourceEvent.type === "mouseup") {
          state.svg.dispatch('zoom', { detail: {
            zoomX: selection,
            zoomY: state.zoomY
          }});
        }
      });

    // Build dom
    const brusher = state.overviewAreaSvg.append('g').attr('class', 'brusher-margins');
    brusher.append('rect').attr('class', 'grid-background');
    brusher.append('g').attr('class', 'x-grid');
    brusher.append('g').attr('class', 'x-axis');
    brusher.append('g').attr('class', 'brush');
          
    //state.svg.on('zoomScent', function() {

    //  const zoomX = d3.event.detail.zoomX;

    //  if (!zoomX) return;

    //  // Out of overview bounds > extend it
    //  if (zoomX[0]<state.overviewArea.domainRange()[0] || zoomX[1]>state.overviewArea.domainRange()[1]) {
    //    console.log("can this happen?");
    //    state.overviewArea.domainRange([
    //      new Date(Math.min(zoomX[0], state.overviewArea.domainRange()[0])),
    //      new Date(Math.max(zoomX[1], state.overviewArea.domainRange()[1]))
    //    ]);
    //  }

    //  state.overviewArea.currentSelection(zoomX);

    //  console.log("on ZoomScent");
    //});
  }

  // Procedure: _make_timeline_axes
  function _make_timeline_axes() {  
    //console.log("making the axes ...");
    const axes = state.svg.append('g').attr('class', 'axes');
    axes.append('g').attr('class', 'x-axis');
    axes.append('g').attr('class', 'x-grid');
    axes.append('g').attr('class', 'y-axis');
    axes.append('g').attr('class', 'grp-axis');

    state.yAxis.scale(state.yScale).tickSize(0);
    state.grpAxis.scale(state.grpScale).tickSize(0);
  }

  // Procedure: _make_timeline_legend
  function _make_timeline_legend() {

    //console.log("making the legend ...");

    // add a reset text
    state.resetBtn = state.svg.append('text')
      .attr('class', 'reset-zoom-btn')
      .text('Reset Zoom')
      .on('click' , function() {
        //console.log("ON 'click': reset btn");
        state.svg.dispatch('resetZoom');
      });
    
    // add a legend group
    state.zScale = d3.scaleOrdinal()
      .domain(['static', 'dynamic', 'cudaflow', 'condition', 'module'])
      .range(['#4682b4', '#FF7F0E', '#6A0DAD', '#41A317', '#0000FF']);

    state.zGroup = state.svg.append('g')
                     .attr('class', 'legend');
    state.zWidth = (state.width-state.leftMargin-state.rightMargin)*3/4;
    state.zHeight = state.topMargin*0.8;

    const binWidth = state.zWidth / state.zScale.domain().length;

    //console.log(binWidth)

    let slot = state.zGroup.selectAll('.z-slot')
      .data(state.zScale.domain());

    slot.exit().remove();

    const newslot = slot.enter()
      .append('g')
      .attr('class', 'z-slot');

    newslot.append('rect')
      .attr('y', 0)
      .attr('rx', 0)
      .attr('ry', 0)
      .attr('stroke-width', 0);

    newslot.append('text')
      .style('text-anchor', 'middle')
      .style('dominant-baseline', 'central');

    // Update
    slot = slot.merge(newslot);

    slot.select('rect')
      .attr('width', binWidth)
      .attr('height', state.zHeight)
      .attr('x', (d, i) => binWidth*i)
      .attr('fill', d => state.zScale(d));

    slot.select('text')
      .text(d => d)
      .attr('x', (d, i) => binWidth*(i+.5))
      .attr('y', state.zHeight*0.5)
      .style('fill', '#FFFFFF');
  }

  // Procedure: _make_timeline_graph
  function _make_timeline_graph() {

    //console.log("making the graph ...");

    state.graph = state.svg.append('g');

    state.graph.on('mousedown', function() {

      //console.log("ON 'mousedown'");

      if (d3.select(window).on('mousemove.zoomRect')!=null) // Selection already active
        return;

      const e = this;

      if (d3.mouse(e)[0]<0 || d3.mouse(e)[0] > state.graphW || 
          d3.mouse(e)[1]<0 || d3.mouse(e)[1] > state.graphH)
        return;

      state.disableHover=true;

      const rect = state.graph.append('rect')
        .attr('class', 'chart-zoom-selection');

      const startCoords = d3.mouse(e);

      d3.select(window)
        .on('mousemove.zoomRect', function() {

          //console.log("ON 'mousemove'");

          d3.event.stopPropagation();
          const newCoords = [
            Math.max(0, Math.min(state.graphW, d3.mouse(e)[0])),
            Math.max(0, Math.min(state.graphH, d3.mouse(e)[1]))
          ];

          rect.attr('x', Math.min(startCoords[0], newCoords[0]))
            .attr('y', Math.min(startCoords[1], newCoords[1]))
            .attr('width', Math.abs(newCoords[0] - startCoords[0]))
            .attr('height', Math.abs(newCoords[1] - startCoords[1]));

          state.overviewAreaSelection = [startCoords[0], newCoords[0]]
                                          .sort(d3.ascending)
                                          .map(state.xScale.invert);
          //zoomY = [startCoords[1], newCoords[1]].sort(d3.ascending).map(d =>
          //  state.yScale.domain().indexOf(state.yScale.invert(d))
          //  + ((state.zoomY && state.zoomY[0])?state.zoomY[0]:0)
          //);

          _render_overview_area();

          //state.svg.dispatch('zoomScent', { detail: {
          //  zoomX: [startCoords[0], newCoords[0]].sort(d3.ascending).map(state.xScale.invert),
          //  zoomY: [startCoords[1], newCoords[1]].sort(d3.ascending).map(d =>
          //    state.yScale.domain().indexOf(state.yScale.invert(d))
          //    + ((state.zoomY && state.zoomY[0])?state.zoomY[0]:0)
          //  )
          //}});
        })
        .on('mouseup.zoomRect', function() {

          //console.log("ON 'mouseup'");

          d3.select(window).on('mousemove.zoomRect', null).on('mouseup.zoomRect', null);
          d3.select('body').classed('stat-noselect', false);
          rect.remove();
          state.disableHover=false;

          const endCoords = [
            Math.max(0, Math.min(state.graphW, d3.mouse(e)[0])),
            Math.max(0, Math.min(state.graphH, d3.mouse(e)[1]))
          ];

          if (startCoords[0]==endCoords[0] && startCoords[1]==endCoords[1]) {
            //console.log("no change");
            return;
          }

          //console.log("coord", endCoords);

          const newDomainX = [startCoords[0], endCoords[0]].sort(d3.ascending).map(state.xScale.invert);

          const newDomainY = [startCoords[1], endCoords[1]].sort(d3.ascending).map(d =>
            state.yScale.domain().indexOf(state.yScale.invert(d))
            + ((state.zoomY && state.zoomY[0])?state.zoomY[0]:0)
          );
          
          state.svg.dispatch('zoom', { detail: {
            zoomX: newDomainX,
            zoomY: newDomainY
          }});
        }, true);

      d3.event.stopPropagation();
    });


  }

  // Procedure: _make_timeline_tooltips
  function _make_timeline_tooltips() {

    //console.log("making the tooltips ...");
    
    // group tooltips 
    state.groupTooltip = d3.tip()
         .attr('class', 'timelines-chart-tooltip')
         .direction('w')
         .offset([0, 0])
         .html(d => {
           const leftPush = (d.hasOwnProperty('timeRange') ?
                            state.xScale(d.timeRange[0]) : 0);
           const topPush = (d.hasOwnProperty('label') ?
                            state.grpScale(d.group) - state.yScale(d.group+'+&+'+d.label) : 0 );
           state.groupTooltip.offset([topPush, -leftPush]);
           return d.group;
         });

    state.svg.call(state.groupTooltip);

    // label tooltips
    state.lineTooltip = d3.tip()
         .attr('class', 'timelines-chart-tooltip')
         .direction('e')
         .offset([0, 0])
         .html(d => {
           const rightPush = (d.hasOwnProperty('timeRange') ? 
                              state.xScale.range()[1]-state.xScale(d.timeRange[1]) : 0);
           state.lineTooltip.offset([0, rightPush]);
           return d.label;
         });

    state.svg.call(state.lineTooltip);
    
    // segment tooltips
    state.segmentTooltip = d3.tip()
      .attr('class', 'timelines-chart-tooltip')
      .direction('s')
      .offset([5, 0])
      .html(d => {
        return `Type: ${d.val}<br>
              Name: ${d.name}<br>
              Time: [${d.timeRange}]<br>
              Span: ${d.timeRange[1]-d.timeRange[0]}`;
      });

    state.svg.call(state.segmentTooltip);
  }
        
  // Proecedure: _make_timeline_events      
  function _make_timeline_events() {

    //console.log("making dom events ...");

    state.svg.on('zoom', function() {

      const evData = d3.event.detail;   // passed custom parameters 
      const zoomX = evData.zoomX;
      const zoomY = evData.zoomY;
      //const redraw = (evData.redraw == null) ? true : evData.redraw;
      
      console.assert((zoomX && zoomY));
      //console.log("ON 'zoom'");

      update(zoomX, zoomY);
      
      // exposed to user
      //if (state.onZoom) {
      //  state.onZoom(state.zoomX, state.zoomY);
      //}
    });

    state.svg.on('resetZoom', function() {
      //console.log("ON resetZoom");
      update([state.minX, state.maxX], [null, null]);
      //if (state.onZoom) state.onZoom(null, null);
    });
  }

  // Procedure: _apply_filters
  function _apply_filters() {

    // Flat data based on segment length
    //state.flatData = (state.minSegmentDuration>0
    //  ? state.completeFlatData.filter(d => (d.timeRange[1]-d.timeRange[0]) >= state.minSegmentDuration)
    //  : state.completeFlatData
    //);
    state.flatData = state.completeFlatData;
    
    console.assert(state.zoomY);

    // zoomY
    //if (state.zoomY == null || state.zoomY==[null, null]) {
    if(state.zoomY == null || (state.zoomY[0] == null && state.zoomY[1] == null)) {
      //console.log("use all y");
      state.structData = state.completeStructData;
      state.nLines = state.totalNLines;
      //for (let i=0, len=state.structData.length; i<len; i++) {
      //  state.nLines += state.structData[i].lines.length;
      //}
      //console.log(state.nLines, state.totalNLines);
      return;
    }

    //console.log("filtering struct Data on ", state.zoomY[0], state.zoomY[1]);

    state.structData = [];
    const cntDwn = [state.zoomY[0] == null ? 0 : state.zoomY[0]]; // Initial threshold
    cntDwn.push(Math.max(
      0, (state.zoomY[1]==null ? state.totalNLines : state.zoomY[1]+1)-cntDwn[0])
    ); // Number of lines

    state.nLines = cntDwn[1];
    for (let i=0, len=state.completeStructData.length; i<len; i++) {

      let validLines = state.completeStructData[i].lines;

      //if(state.minSegmentDuration>0) {  // Use only non-filtered (due to segment length) groups/labels
      //  if (!state.flatData.some(d => d.group == state.completeStructData[i].group)) {
      //    continue; // No data for this group
      //  }

      //  validLines = state.completeStructData[i].lines
      //    .filter(d => state.flatData.some(dd =>
      //      dd.group == state.completeStructData[i].group && dd.label == d
      //    )
      //  );
      //}

      if (cntDwn[0]>=validLines.length) { // Ignore whole group (before start)
        cntDwn[0]-=validLines.length;
        continue;
      }

      const groupData = {
        group: state.completeStructData[i].group,
        lines: null
      };

      if (validLines.length-cntDwn[0]>=cntDwn[1]) {  // Last (or first && last) group (partial)
        groupData.lines = validLines.slice(cntDwn[0],cntDwn[1]+cntDwn[0]);
        state.structData.push(groupData);
        cntDwn[1]=0;
        break;
      }

      if (cntDwn[0]>0) {  // First group (partial)
        groupData.lines = validLines.slice(cntDwn[0]);
        cntDwn[0]=0;
      } else {  // Middle group (full fit)
        groupData.lines = validLines;
      }

      state.structData.push(groupData);
      cntDwn[1]-=groupData.lines.length;
    }

    state.nLines-=cntDwn[1];

    //console.log("filtered lines:", state.nLines);
  }


  function _adjust_dimensions() {
    
    //console.log("adjusting up dimensions ... nLines =", state.nLines);

    state.graphW = state.width - state.leftMargin - state.rightMargin;
    //state.graphH = d3.min([
    //  state.nLines*state.maxLineHeight, state.maxHeight-state.topMargin-state.bottomMargin
    //]);
    state.graphH = state.nLines*state.maxLineHeight;
    state.height = state.graphH + state.topMargin + state.bottomMargin;

    //console.log("transition to", state.width, state.height, " graph", state.graphH, state.graphW);

    state.svg.transition().duration(state.transDuration)
      .attr('width', state.width)
      .attr('height', state.height);

    state.graph.attr('transform', `translate(${state.leftMargin}, ${state.topMargin})`);

    //state.overviewArea
    //    .width(state.width * 0.8)
    //    .height(state.overviewHeight + state.overviewArea.margins().top + state.overviewArea.margins().bottom);
    //}
  }

  function _adjust_xscale() {

    console.assert(state.zoomX[0]);
    console.assert(state.zoomX[1]);

    //state.zoomX[0] = state.zoomX[0] || d3.min(state.flatData, d => d.timeRange[0]);
    //state.zoomX[1] = state.zoomX[1] || d3.max(state.flatData, d => d.timeRange[1]);

    //console.log("adjusting xscale to", state.zoomX);

    //state.xScale = (state.useUtc ? d3ScaleUtc : d3ScaleTime)()
    state.xScale.domain(state.zoomX)
                .range([0, state.graphW])
                .clamp(true);

    if (state.overviewArea) {
      state.overviewArea
        .scale(state.xScale.copy());
        //.tickFormat(state.xTickFormat);
    }
  }

  // Procedure: _adjust_yscale
  function _adjust_yscale() {

    let labels = [];
    for (let i= 0, len=state.structData.length; i<len; i++) {
      labels = labels.concat(state.structData[i].lines.map(function (d) {
        return state.structData[i].group + '+&+' + d
      }));
    }

    //console.log("adjusting yscale to", labels);

    state.yScale.domain(labels);

    
    //console.log(state.graphH/labels.length*0.5, state.graphH*(1-0.5/labels.length));

    state.yScale.range([state.graphH/labels.length*0.5, state.graphH*(1-0.5/labels.length)]);
  }
      
  // Procedure: _adjust_grpscale
  function _adjust_grpscale() {

    //console.log("adjusting group domain", state.structData.map(d => d.group));

    state.grpScale.domain(state.structData.map(d => d.group));

    let cntLines = 0;

    state.grpScale.range(state.structData.map(d => {
      const pos = (cntLines + d.lines.length/2)/state.nLines*state.graphH;
      cntLines += d.lines.length;
      return pos;
    }));
  }

  // Procedure: _adjust_legend
  function _adjust_legend() {

    //console.log("adjusting legend ...");

    state.resetBtn
      .transition().duration(state.transDuration)
        .attr('x', state.leftMargin + state.graphW*.99)
        .attr('y', state.topMargin *.8);
    
    state.zGroup
      .transition().duration(state.transDuration)
        .attr('transform', `translate(${state.leftMargin}, ${state.topMargin * .1})`);
  }

  // Procedure: _render_axes
  function _render_axes() {

    state.svg.select('.axes')
      //.attr('transform', 'translate(' + state.leftMargin + ',' + state.topMargin + ')');
      .attr('transform', `translate(${state.leftMargin}, ${state.topMargin})`);

    // X
    //const nXTicks = Math.max(2, Math.min(12, Math.round(state.graphW * 0.012)));
    const nXTicks = num_xticks(state.graphW);

    //console.log("rendering axes nXTicks =", nXTicks);

    state.xAxis
      .scale(state.xScale)
      .ticks(nXTicks);

    state.xGrid
      .scale(state.xScale)
      .ticks(nXTicks)
      .tickFormat('');

    state.svg.select('g.x-axis')
      .style('stroke-opacity', 0)
      .style('fill-opacity', 0)
      .attr('transform', 'translate(0,' + state.graphH + ')')
      .transition().duration(state.transDuration)
        .call(state.xAxis)
        .style('stroke-opacity', 1)
        .style('fill-opacity', 1);

    /* Angled x axis labels
     state.svg.select('g.x-axis').selectAll('text')
     .style('text-anchor', 'end')
     .attr('transform', 'translate(-10, 3) rotate(-60)');
     */

    state.xGrid.tickSize(state.graphH);
    state.svg.select('g.x-grid')
      .attr('transform', 'translate(0,' + state.graphH + ')')
      .transition().duration(state.transDuration)
      .call(state.xGrid);

    if (
      state.dateMarker &&
      state.dateMarker >= state.xScale.domain()[0] &&
      state.dateMarker <= state.xScale.domain()[1]
    ) {
      state.dateMarkerLine
        .style('display', 'block')
        .transition().duration(state.transDuration)
          .attr('x1', state.xScale(state.dateMarker) + state.leftMargin)
          .attr('x2', state.xScale(state.dateMarker) + state.leftMargin)
          .attr('y1', state.topMargin + 1)
          .attr('y2', state.graphH + state.topMargin);
    } else {
      state.dateMarkerLine.style('display', 'none');
    }

    // Y
    const fontVerticalMargin = 0.6;
    const labelDisplayRatio = Math.ceil(
      state.nLines*state.minLabelFont/Math.sqrt(2)/state.graphH/fontVerticalMargin
    );
    const tickVals = state.yScale.domain().filter((d, i) => !(i % labelDisplayRatio));
    let fontSize = Math.min(14, state.graphH/tickVals.length*fontVerticalMargin*Math.sqrt(2));
    let maxChars = Math.ceil(state.rightMargin/(fontSize/Math.sqrt(2)));

    state.yAxis.tickValues(tickVals);
    state.yAxis.tickFormat(d => reduceLabel(d.split('+&+')[1], maxChars));
    state.svg.select('g.y-axis')
      .transition().duration(state.transDuration)
        .attr('transform', `translate(${state.graphW}, 0)`)
        .attr('font-size', `${fontSize}px`)
        .call(state.yAxis);

    // Grp
    const minHeight = d3.min(state.grpScale.range(), function (d, i) {
      return i>0 ? d-state.grpScale.range()[i-1] : d*2;
    });

    fontSize = Math.min(14, minHeight*fontVerticalMargin*Math.sqrt(2));
    maxChars = Math.ceil(state.leftMargin/(fontSize/Math.sqrt(2)));
    
    //console.log(minHeight, maxChars);

    state.grpAxis.tickFormat(d => reduceLabel(d, maxChars));
    state.svg.select('g.grp-axis')
      .transition().duration(state.transDuration)
      .attr('font-size', `${fontSize}px`)
      .call(state.grpAxis);

    //// Make Axises clickable
    //if (state.onLabelClick) {
    //  state.svg.selectAll('g.y-axis,g.grp-axis').selectAll('text')
    //    .style('cursor', 'pointer')
    //    .on('click', function(d) {
    //      const segms = d.split('+&+');
    //      state.onLabelClick(...segms.reverse());
    //    });
    //}

    function reduceLabel(label, maxChars) {
      return label.length <= maxChars ? label : (
        label.substring(0, maxChars*2/3)
        + '...'
        + label.substring(label.length - maxChars/3, label.length
      ));
    }
  }

  // Procedure: _render_groups
  function _render_groups() {

    let groups = state.graph.selectAll('rect.series-group').data(state.structData, d => d.group);
    //console.log("rendering groups", groups);
        
    groups.exit()
      .transition().duration(state.transDuration)
      .style('stroke-opacity', 0)
      .style('fill-opacity', 0)
      .remove();
      
    //  fill-opacity: 0.6;
    //  stroke: #808080;
    //  stroke-opacity: 0.2;

    const newGroups = groups.enter().append('rect')
      .attr('class', 'series-group')
      .attr('x', 0)
      .attr('y', 0)
      .attr('height', 0)
      .style('fill', `url(#${state.groupGradId})`)
      .on('mouseover', state.groupTooltip.show)
      .on('mouseout', state.groupTooltip.hide);

    newGroups.append('title')
      .text('click-drag to zoom in');

    groups = groups.merge(newGroups);

    groups.transition().duration(state.transDuration)
      .attr('width', state.graphW)
      .attr('height', function (d) {
        return state.graphH*d.lines.length/state.nLines;
      })
      .attr('y', function (d) {
        return state.grpScale(d.group)-state.graphH*d.lines.length/state.nLines/2;
      });
  }

  // procedure: _render_timelines
  function _render_timelines(maxElems) {

    //console.log("rendering timelines ...");

    if (maxElems == undefined || maxElems < 0) {
      maxElems = null;
    }

    const hoverEnlargeRatio = .4;

    const dataFilter = (d, i) =>
      (maxElems == null || i<maxElems) &&
      (state.grpScale.domain().indexOf(d.group)+1 &&
       d.timeRange[1]>=state.xScale.domain()[0] &&
       d.timeRange[0]<=state.xScale.domain()[1] &&
       state.yScale.domain().indexOf(d.group+'+&+'+d.label)+1);

    state.lineHeight = state.graphH/state.nLines*0.8;

    let timelines = state.graph.selectAll('rect.series-segment').data(
      state.flatData.filter(dataFilter),
      d => d.group + d.label + d.timeRange[0]
    );

    timelines.exit()
      .transition().duration(state.transDuration)
      .style('fill-opacity', 0)
      .remove();

    const newSegments = timelines.enter().append('rect')
      .attr('class', 'series-segment')
      .attr('rx', 1)
      .attr('ry', 1)
      .attr('x', state.graphW/2)
      .attr('y', state.graphH/2)
      .attr('width', 0)
      .attr('height', 0)
      .style('fill', d => state.zColorMap.get(d.val))
      .style('fill-opacity', 0)
      .on('mouseover.groupTooltip', state.groupTooltip.show)
      .on('mouseout.groupTooltip', state.groupTooltip.hide)
      .on('mouseover.lineTooltip', state.lineTooltip.show)
      .on('mouseout.lineTooltip', state.lineTooltip.hide)
      .on('mouseover.segmentTooltip', state.segmentTooltip.show)
      .on('mouseout.segmentTooltip', state.segmentTooltip.hide);

    newSegments
      .on('mouseover', function() {

        if (state.disableHover) {
          return;
        }

        //MoveToFront()(this);

        const hoverEnlarge = state.lineHeight*hoverEnlargeRatio;

        d3.select(this)
          .transition().duration(70)
          .attr('x', function (d) {
            return state.xScale(d.timeRange[0])-hoverEnlarge/2;
          })
          .attr('width', function (d) {
            return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])])+hoverEnlarge;
          })
          .attr('y', function (d) {
            return state.yScale(`${d.group}+&+${d.label}`)-(state.lineHeight+hoverEnlarge)/2;
          })
          .attr('height', state.lineHeight+hoverEnlarge)
          .style('fill-opacity', 1);
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition().duration(250)
          .attr('x', function (d) {
            return state.xScale(d.timeRange[0]);
          })
          .attr('width', function (d) {
            return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])]);
          })
          .attr('y', function (d) {
            return state.yScale(`${d.group}+&+${d.label}`)-state.lineHeight/2;
          })
          .attr('height', state.lineHeight)
          .style('fill-opacity', .8);
      })
      .on('click', function (s) {
        if (state.onSegmentClick)
          state.onSegmentClick(s);
      });

    timelines = timelines.merge(newSegments);

    timelines.transition().duration(state.transDuration)
      .attr('x', function (d) {
        return state.xScale(d.timeRange[0]);
      })
      .attr('width', function (d) {
        return d3.max([1, state.xScale(d.timeRange[1])-state.xScale(d.timeRange[0])]);
      })
      .attr('y', function (d) {
        return state.yScale(`${d.group}+&+${d.label}`)-state.lineHeight/2;
      })
      .attr('height', state.lineHeight)
      .style('fill-opacity', .8);
  }

  function _render_overview_area()  {

    //console.log("rendering overview...")
    
    // domain is not set up yet
    if (state.overviewAreaDomain[0] == null || state.overviewAreaDomain[1] == null) {
      return;
    }

    const brushWidth = state.graphW;
    const brushHeight = 20;
    const nXTicks = num_xticks(brushWidth);

    //console.log("brush ", brushWidth, brushHeight);

    state.overviewAreaScale
      .domain(state.overviewAreaDomain)
      .range([0, brushWidth]);

    state.overviewAreaXAxis
      .scale(state.overviewAreaScale)
      .ticks(nXTicks);

    state.overviewAreaXGrid
      .scale(state.overviewAreaScale)
      .tickSize(-brushHeight);

    state.overviewAreaSvg
      .attr('width', state.width)
      .attr('height', brushHeight + state.overviewAreaTopMargin
                                  + state.overviewAreaBottomMargin);

    state.overviewAreaSvg.select('.brusher-margins')
      .attr('transform', `translate(${state.leftMargin}, ${state.overviewAreaTopMargin})`);

    state.overviewAreaSvg.select('.grid-background')
      //.attr('transform', `translate(${state.leftMargin},${})`)
      .attr('width', brushWidth)
      .attr('height', brushHeight);

    state.overviewAreaSvg.select('.x-grid')
      .attr('transform', `translate(0, ${brushHeight})`)
      .call(state.overviewAreaXGrid);

    state.overviewAreaSvg.select('.x-axis')
      .attr("transform", `translate(0, ${brushHeight})`)
      .call(state.overviewAreaXAxis)
      .selectAll('text').attr('y', 8);

    state.overviewAreaSvg.select('.brush')
      .call(state.overviewAreaBrush.extent([[0, 0], [brushWidth, brushHeight]]))
      .call(state.overviewAreaBrush.move, state.overviewAreaSelection.map(state.overviewAreaScale));
  }

  // ----------------------------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------------------------
  function num_xticks(W) {
    return Math.max(2, Math.min(12, Math.round(W * 0.012)));
  }

  // ----------------------------------------------------------------------------
  // ----------------------------------------------------------------------------

  // Example: Matrix multiplication
  $('#tfp_matmul').on('click', function() {
    tfp_render_matmul();
  });

  $('#tfp_kmeans').on('click', function() {
    tfp_render_kmeans();
  });

  $('#tfp_inference').on('click', function() {
    tfp_render_inference();
  });

  $('#tfp_dreamplace').on('click', function() {
    tfp_render_dreamplace();
  });

  // textarea changer event
  $('#tfp_textarea').on('input propertychange paste', function() {

    if($(this).data('timeout')) {
      clearTimeout($(this).data('timeout'));
    }

    $(this).data('timeout', setTimeout(()=>{
      
      var text = $('#tfp_textarea').val().trim();
      
      $('#tfp_textarea').removeClass('is-invalid');

      if(!text) {
        return;
      }
      
      try {
        var json = JSON.parse(text);
        console.log(json);
        feed(json);
      }
      catch(e) {
        $('#tfp_textarea').addClass('is-invalid');
        console.log(e);
      }

    }, 3000));
  });

  // render the timeline from a parsed json
  //function tfp_render_timeline((json) {
    // clear the existing timeline
    //document.getElementById("MyDiv").innerHTML = "";
    //$('#tfp_timeline').html('');
    //feed(json);
  //}

  // render default data
  function tfp_render_simple() {
    feed(simple);
    $('#tfp_textarea').text(JSON.stringify(simple, null, 2));
  }

  function tfp_render_matmul() {
    feed(matmul);
    $('#tfp_textarea').text(JSON.stringify(matmul));
  }

  function tfp_render_kmeans() {
    feed(kmeans);
    $('#tfp_textarea').text(JSON.stringify(kmeans));
  }

  function tfp_render_inference() {
    feed(inference);
    $('#tfp_textarea').text(JSON.stringify(inference));
  }

  function tfp_render_dreamplace() {
    feed(dreamplace);
    $('#tfp_textarea').text(JSON.stringify(dreamplace));
  }

  // ----------------------------------------------------------------------------

  // DOM objects
  make_timeline_structure(document.getElementById('tfp_timeline'));

  tfp_render_simple();

}());
