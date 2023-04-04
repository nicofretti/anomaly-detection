import json

import dash
import numpy as np
from dash import dcc, html, Output, Input, callback
from flask import Flask, request
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import image as mpimg
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

SERVER = Flask(__name__)

APP = dash.Dash(
    __name__,
    server=SERVER,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_scripts=[
        # Tailwind CSS
        "https://tailwindcss.com/",
        {
            "src": "https://cdn.tailwindcss.com"
        }
    ],
    external_stylesheets=[
        # Font Awesome
        {
            'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
            'rel': 'stylesheet',
            'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
            'crossorigin': 'anonymous'
        }
    ]
)

MAP_POSITION = [

    [0.0, 0.0, False],
    [0.0, 0.0, False],
    [0.0, 0.0, False],
    [-0.005556958561722114, -0.0004035011679900391, False],
    [-0.01331837071493891, -0.0007867760549454816, False],
    [-0.01331837071493891, -0.0007867760549454816, False],
    [-0.03195633052465707, -0.00148230452236231, False],
    [-0.04595813469225185, -0.00205019862584388, False],
    [-0.04595813469225185, -0.00205019862584388, False],
    [-0.05683935191828371, -0.0022138013173232385, False],
    [-0.07491035168917892, -0.002889917446538881, False],
    [-0.07491035168917892, -0.002889917446538881, False],
    [-0.1020523362332405, -0.004612601839527719, False],
    [-0.1262268586992048, -0.005586831597694852, False],
    [-0.1262268586992048, -0.005586831597694852, False],
    [-0.1262268586992048, -0.005586831597694852, False],
    [-0.16854866893736975, -0.0076076612820464296, False],
    [-0.18225890115240262, -0.008380434206487486, False],
    [-0.18225890115240262, -0.008380434206487486, False],
    [-0.21313660195777973, -0.00949584198889375, False],
    [-0.21313660195777973, -0.00949584198889375, False],
    [-0.2485629811768605, -0.011185541042416813, False],
    [-0.28816456696737147, -0.020121522335868863, False],
    [-0.28816456696737147, -0.020121522335868863, False],
    [-0.3200382090415278, -0.021749642012546005, False],
    [-0.3534320803717109, -0.023114123988937113, False],
    [-0.3534320803717109, -0.023114123988937113, False],
    [-0.3534320803717109, -0.023114123988937113, False],
    [-0.4193588422482971, -0.026188577508963057, False],
    [-0.43579513021741356, -0.02691300681563477, False],
    [-0.43579513021741356, -0.02691300681563477, False],
    [-0.4669241684992874, -0.028203618475708597, False],
    [-0.490389588609632, -0.014962505325990694, False],
    [-0.490389588609632, -0.014962505325990694, False],
    [-0.5149241386147799, -0.016177279449245546, False],
    [-0.5149241386147799, -0.016177279449245546, False],
    [-0.5361891600433227, -0.01767075813901653, False],
    [-0.5328900896815832, -0.01756800837305253, False],
    [-0.5328900896815832, -0.01756800837305253, False],
    [-0.5267859798725724, -0.01722307571695797, False],
    [-0.5267859798725724, -0.01722307571695797, False],
    [-0.5315742008978585, -0.01948408936878529, False],
    [-0.5316192169778834, -0.019533802248587584, False],
    [-0.5306323805489802, -0.02086693770507131, False],
    [-0.5295320811536072, -0.027999562170153508, False],
    [-0.5295320811536072, -0.027999562170153508, False],
    [-0.5291451341615675, -0.046984229921676585, False],
    [-0.5282886390437277, -0.06959113801837644, False],
    [-0.5282886390437277, -0.06959113801837644, False],
    [-0.5272118944438741, -0.08843968625091642, False],
    [-0.5272118944438741, -0.08843968625091642, False],
    [-0.5263347562663349, -0.10721238135979874, False],
    [-0.5260899381349186, -0.12615684801842222, False],
    [-0.5260899381349186, -0.12615684801842222, False],
    [-0.5254850439670838, -0.1453016444215942, False],
    [-0.5277528841263722, -0.1642803384774922, False],
    [-0.5277528841263722, -0.1642803384774922, False],
    [-0.5303909505351, -0.18202893421644536, False],
    [-0.5333791101751643, -0.20095179897004833, False],
    [-0.5333791101751643, -0.20095179897004833, False],
    [-0.5402718559029612, -0.22716129228380602, False],
    [-0.5404491298903012, -0.22767127678618443, True],
    [-0.5515842782970807, -0.2323610344895406, True],
    [-0.5547041820372801, -0.24162402730993077, True],
    [-0.561875873335211, -0.2595870400403031, True],
    [-0.565146617927058, -0.2683815552315927, True],
    [-0.5692873851500543, -0.276723956634512, True],
    [-0.5783866995002435, -0.2935032054181681, False],
    [-0.5832407667163776, -0.3021844276291191, False],
    [-0.593958779351628, -0.3211102011979461, False],
    [-0.593958779351628, -0.3211102011979461, False],
    [-0.5978677894231278, -0.3435557655184679, False],
    [-0.5978677894231278, -0.3435557655184679, False],
    [-0.5957520094087408, -0.3705961040847518, False],
    [-0.594756651363476, -0.3958606563490133, False],
    [-0.594756651363476, -0.3958606563490133, False],
    [-0.5973007338687303, -0.4206844163115099, False],
    [-0.6038729918487568, -0.4442411754376151, False],
    [-0.6038729918487568, -0.4442411754376151, False],
    [-0.6017905461605629, -0.47072352669397616, False],
    [-0.5939429204870855, -0.5029269431346959, False],
    [-0.5939429204870855, -0.5029269431346959, False],
    [-0.5853930541636599, -0.5369445613236787, False],
    [-0.5853930541636599, -0.5369445613236787, False],
    [-0.5780783697211019, -0.5677262998237924, False],
    [-0.5722807837012392, -0.5990560142104397, False],
    [-0.5722807837012392, -0.5990560142104397, False],
    [-0.5706490189765593, -0.6304663371704005, False],
    [-0.5708154054506097, -0.6538783556780783, False],
    [-0.5708154054506097, -0.6538783556780783, False],
    [-0.578447800282824, -0.6855901555481525, False],
    [-0.5858339628092125, -0.7171088969651291, False],
    [-0.5858339628092125, -0.7171088969651291, False],
    [-0.5850312219524186, -0.7477264541441863, False],
    [-0.5850312219524186, -0.7477264541441863, False],
    [-0.5819014419492642, -0.7792793812919778, False],
    [-0.5860163339155467, -0.8107432956957303, False],
    [-0.5860163339155467, -0.8107432956957303, False],
    [-0.5919191157704279, -0.8416166768474206, False],
    [-0.6042954329748859, -0.875926380766058, False],
    [-0.6042954329748859, -0.875926380766058, False],
    [-0.6028532681013657, -0.9113834074205995, False],
    [-0.6028532681013657, -0.9113834074205995, False],
    [-0.5986696545042852, -0.9503794381768764, False],
    [-0.5980804081728169, -0.9883155620731368, False],
    [-0.5980804081728169, -0.9883155620731368, False],
    [-0.5980804081728169, -0.9883155620731368, False],
    [-0.6117042575622554, -1.0561755556758288, False],
    [-0.6165126765078118, -1.0699915638038582, False],
    [-0.6165126765078118, -1.0699915638038582, False],
    [-0.6255214302338173, -1.0884366565383559, False],
    [-0.6212757245197108, -1.079098579901934, False],
    [-0.6212757245197108, -1.079098579901934, False],
    [-0.6228590714321266, -1.0787372485679023, False],
    [-0.625979551028304, -1.0852681717862267, False],
    [-0.625979551028304, -1.0852681717862267, False],
    [-0.633533573019932, -1.1000368222867885, False],
    [-0.633533573019932, -1.1000368222867885, False],
    [-0.653016838911941, -1.1293421790696403, True],
    [-0.6907180927836208, -1.1620697779167835, True],
    [-0.6907180927836208, -1.1620697779167835, True],
    [-0.750974243199929, -1.226216507436258, True],
    [-0.7675626062582449, -1.2472358038405398, True],
    [-0.7675626062582449, -1.2472358038405398, True],
    [-0.7950158860955309, -1.2847108011959212, True],
    [-0.8071485269941808, -1.2978636870932938, True],
    [-0.8071485269941808, -1.2978636870932938, True],
    [-0.7997596623135356, -1.2898083601923367, True],
    [-0.7997596623135356, -1.2898083601923367, True],
    [-0.7962962578162495, -1.287058677499236, True],
    [-0.8002874011461639, -1.2921743324556059, True],
    [-0.8002874011461639, -1.2921743324556059, True],
    [-0.8002874011461639, -1.2921743324556059, True],
    [-0.8006740692781081, -1.283682970907209, True],
    [-0.8006740692781081, -1.283682970907209, True],
    [-0.8182244389673224, -1.2559525736511088, True],
    [-0.8231990303598494, -1.244760624635599, True],
    [-0.8231990303598494, -1.244760624635599, True],
    [-0.8362134006498313, -1.2152430403883718, True],
    [-0.8362134006498313, -1.2152430403883718, True],
    [-0.8436105875951682, -1.19353405785982, True],
    [-0.8545832419633687, -1.1680459882093512, True],
    [-0.8545832419633687, -1.1680459882093512, True],
    [-0.8545832419633687, -1.1680459882093512, True],
    [-0.8798890357341588, -1.118882622627837, True],
    [-0.8798890357341588, -1.118882622627837, True],
    [-0.90768276496115, -1.0833223861820755, True],
    [-0.90768276496115, -1.0833223861820755, True],
    [-0.9441573272615366, -1.0582907063286582, True],
    [-0.950192165861078, -1.056188623278129, True],
    [-0.9704452641678875, -1.0505290342914375, True],
    [-1.0123110697163646, -1.0464968342617817, True],
    [-1.0123110697163646, -1.0464968342617817, True],
    [-1.0479059936294912, -1.0536237614899986, True],
    [-1.0479059936294912, -1.0536237614899986, True],
    [-1.081836042132311, -1.0727484203104214, True],
    [-1.092829566897405, -1.0743380299890424, True],
    [-1.092829566897405, -1.0743380299890424, True],
    [-1.092829566897405, -1.0743380299890424, True],
    [-1.1121584284874868, -1.0800357144296866, True],
    [-1.1121584284874868, -1.0800357144296866, True],
    [-1.1249688185926146, -1.0844382608235663, True],
    [-1.1286515089295959, -1.0860135382032792, True],
    [-1.1286515089295959, -1.0860135382032792, True],
    [-1.132462303727841, -1.0894969798897853, True],
    [-1.132462303727841, -1.0894969798897853, True],
    [-1.1365460209178608, -1.0949059897878506, True],
    [-1.1401561036264538, -1.096413812256786, True],
    [-1.1401561036264538, -1.096413812256786, True],
    [-1.1401561036264538, -1.096413812256786, True],
    [-1.1370031604428763, -1.095145845817244, True],
    [-1.1370031604428763, -1.095145845817244, True],
    [-1.1380573951686146, -1.0957712274790379, True],
    [-1.1380573951686146, -1.0957712274790379, True],
    [-1.1304908503197635, -1.1084893749209095, True],
    [-1.128925021354719, -1.111405536965853, True],
    [-1.1240707929957647, -1.1212330270529063, True],
    [-1.122860058326868, -1.1263304078755445, True],
    [-1.122860058326868, -1.1263304078755445, True],
    [-1.1242309362666862, -1.1201403230626223, True],
    [-1.1251089139368395, -1.119974049632229, True],
    [-1.1251089139368395, -1.119974049632229, True],
    [-1.1251089139368395, -1.119974049632229, True],
    [-1.124153943926299, -1.1218678473987318, True],
    [-1.124153943926299, -1.1218678473987318, True],
    [-1.114445350408265, -1.1409744086101368, True],
    [-1.1125644655467144, -1.1442007740556859, True],
    [-1.1125644655467144, -1.1442007740556859, True],
    [-1.087592432313326, -1.165689614973269, True],
    [-1.073616662073769, -1.166249889645542, True],
    [-1.073616662073769, -1.166249889645542, True],
    [-1.0419274613156162, -1.176938198885799, True],
    [-1.0351699547202897, -1.1910252077250938, True],
    [-1.0351699547202897, -1.1910252077250938, True],
    [-1.0351699547202897, -1.1910252077250938, True],
    [-1.0557416898118683, -1.213388811824758, True],
    [-1.0557416898118683, -1.213388811824758, True],
    [-1.0465844992415492, -1.205173260140264, True],
    [-1.0466225070630397, -1.2053637912936424, True],
    [-1.0466225070630397, -1.2053637912936424, True],
    [-1.0500933836451454, -1.2108005985116408, True],
    [-1.0477776918876969, -1.2109377083616633, True],
    [-1.0477776918876969, -1.2109377083616633, True],
    [-1.033630025691041, -1.2179004407330714, True],
    [-1.0015106865580101, -1.2402985078130304, True],
    [-1.0015106865580101, -1.2402985078130304, True],
    [-1.0015106865580101, -1.2402985078130304, True],
    [-0.9393757985371957, -1.3022579979800861, True],
    [-0.9393757985371957, -1.3022579979800861, True],
    [-0.9270601141889824, -1.3299945600591923, True],
    [-0.9270601141889824, -1.3299945600591923, True],
    [-0.920457549380303, -1.3346133292595175, True],
    [-0.9198214146840998, -1.3355301992919466, True],
    [-0.9198214146840998, -1.3355301992919466, True],
    [-0.9039266899648751, -1.3645952994026034, True],
    [-0.8958075599544683, -1.3783022383350976, True],
    [-0.8958075599544683, -1.3783022383350976, True],
    [-0.8916370304288866, -1.4027794046628284, True],
    [-0.8916370304288866, -1.4027794046628284, True],
    [-0.8736791799875546, -1.4500704446869657, True],
    [-0.8728151028842249, -1.4521832738104548, True],
    [-0.8728151028842249, -1.4521832738104548, True],
    [-0.8552712070076253, -1.489249487714475, True],
    [-0.8552712070076253, -1.489249487714475, True],
    [-0.8350166369678221, -1.520400469097367, True],
    [-0.8350166369678221, -1.520400469097367, True],
    [-0.8311242759344467, -1.522364180180247, True],
    [-0.8312454683355807, -1.5213776445416227, True],
    [-0.8296477335599549, -1.517742932271002, True],
    [-0.8143506889868618, -1.5042820745662937, True],
    [-0.8143506889868618, -1.5042820745662937, True],
    [-0.7951018582895261, -1.4874333969128395, True],
    [-0.7875607393267577, -1.480762819556303, True],
    [-0.7875607393267577, -1.480762819556303, True],
    [-0.7875607393267577, -1.480762819556303, True],
    [-0.7926485592444489, -1.4879725017560403, True],
    [-0.7926485592444489, -1.4879725017560403, True],
    [-0.7664532079700096, -1.4996760268588658, True],
    [-0.761715429918005, -1.5031414110947312, True],
    [-0.7468687172183396, -1.5155762117834406, True],
    [-0.7169097817113804, -1.5423568951617186, True],
    [-0.7169097817113804, -1.5423568951617186, True],
    [-0.7169097817113804, -1.5423568951617186, True],
    [-0.6802909200689321, -1.5874694217383694, True],
    [-0.675736355855353, -1.5921987352981954, True],
    [-0.675736355855353, -1.5921987352981954, True],
    [-0.675736355855353, -1.5921987352981954, True],
    [-0.6574094403614749, -1.6400515006868899, True],
    [-0.6574094403614749, -1.6400515006868899, True],
    [-0.6460119053737444, -1.6775835387631515, True],
    [-0.6455971135418977, -1.6786589387377489, True],
    [-0.6455971135418977, -1.6786589387377489, True],
    [-0.6305211677095249, -1.7140485223307245, True],
    [-0.6271817095178361, -1.7314152462436438, True],
    [-0.6271817095178361, -1.7314152462436438, True],
    [-0.6247447141124894, -1.7789965053799688, True],
    [-0.6311727067069405, -1.8080574678163572, True],
    [-0.6311727067069405, -1.8080574678163572, True],
    [-0.6311727067069405, -1.8080574678163572, True],
    [-0.6127202979086791, -1.8898457023196715, True],
    [-0.6127202979086791, -1.8898457023196715, True],
    [-0.5947502715265939, -1.9508132007956016, True],
    [-0.5947502715265939, -1.9508132007956016, True],
    [-0.5874007711292389, -1.9708917676288173, True],
    [-0.592702712127725, -1.9898767972599898, True],
    [-0.5862559970991522, -2.010350530255743, True],
    [-0.5738881331135498, -2.051101495416801, True],
    [-0.5738881331135498, -2.051101495416801, True],
    [-0.5666143350619718, -2.091825936701456, True],
    [-0.5666143350619718, -2.091825936701456, True],
    [-0.5666654871759645, -2.135139369707405, True],
    [-0.568086359516451, -2.1787473173863985, True],
    [-0.568086359516451, -2.1787473173863985, True],
    [-0.568086359516451, -2.1787473173863985, True],
    [-0.565009635433542, -2.240363250296115, True],
    [-0.565009635433542, -2.240363250296115, True],
    [-0.5578729706997633, -2.304865764756056, True],
    [-0.5563533952816117, -2.3259632568790645, True],
    [-0.5563533952816117, -2.3259632568790645, True],
    [-0.5525274634306406, -2.3677424246444865, True],
    [-0.5525274634306406, -2.3677424246444865, True],
    [-0.5630858135212514, -2.396906854219962, True],
    [-0.555697361954819, -2.438762286078658, True],
    [-0.555697361954819, -2.438762286078658, True],
    [-0.5456804458057573, -2.4798658793142456, True],
    [-0.5456804458057573, -2.4798658793142456, True],
    [-0.5289943786539283, -2.5615585959815164, True],
    [-0.5283806415957114, -2.564544147503117, True],
    [-0.5283806415957114, -2.564544147503117, True],
    [-0.5390311400750726, -2.611862117305006, True],
    [-0.5390311400750726, -2.611862117305006, True],
    [-0.5274591351763586, -2.675470690762557, True],
    [-0.5245804126636803, -2.6972656512991975, True],
    [-0.5245804126636803, -2.6972656512991975, True],
    [-0.5200017632129116, -2.742345735301182, False],
    [-0.5152607454077979, -2.788069436856915, False],
    [-0.5152607454077979, -2.788069436856915, False],
    [-0.5152607454077979, -2.788069436856915, False],
    [-0.5059741425748633, -2.8979685198076375, False],
    [-0.5059741425748633, -2.8979685198076375, False],
    [-0.5034814344485529, -2.919796937628102, False],
    [-0.49776495712833724, -2.9642854196360435, False],
    [-0.49776495712833724, -2.9642854196360435, False],
    [-0.4904462275630873, -3.032853345598422, False],
    [-0.5024925140861064, -3.054333991072867, False],
    [-0.5024925140861064, -3.054333991072867, False],
    [-0.49757726495346744, -3.099474233281531, False],
    [-0.4923771802672684, -3.144586185165103, False],
    [-0.4923771802672684, -3.144586185165103, False],
    [-0.4923771802672684, -3.144586185165103, False],
    [-0.4812742542225351, -3.234949391943628, False],
    [-0.4812742542225351, -3.234949391943628, False],
    [-0.4584243907537294, -3.3139398145628967, False],
    [-0.45789306765526716, -3.316277713593437, False],
    [-0.45789306765526716, -3.316277713593437, False],
    [-0.44361877276627804, -3.3824788398389805, False],
    [-0.44361877276627804, -3.3824788398389805, False],
    [-0.42875445474884366, -3.4510617898364693, False],
    [-0.4264833868090582, -3.4768186234952845, False],
    [-0.4264833868090582, -3.4768186234952845, False],
    [-0.4165660554971059, -3.5213434445971723, False],
    [-0.4165660554971059, -3.5213434445971723, False],
    [-0.40844492631895724, -3.566656995628554, False],
    [-0.4025815362548516, -3.615730298091699, False],
    [-0.4025815362548516, -3.615730298091699, False],
    [-0.4009664977899786, -3.6782106138805233, False],
    [-0.4009664977899786, -3.6782106138805233, False],
    [-0.39238843497100506, -3.752322673155561, False],
    [-0.3922468638451445, -3.7532671966379487, False],
    [-0.3922468638451445, -3.7532671966379487, False],
    [-0.3812842118057972, -3.829060593882085, False],
    [-0.37660801918961095, -3.8547054686527478, False],
    [-0.37660801918961095, -3.8547054686527478, False],
    [-0.37660801918961095, -3.8547054686527478, False],
    [-0.34922072354155376, -3.950473500303673, False],
    [-0.34922072354155376, -3.950473500303673, False],
    [-0.3321733392943157, -4.025220792922246, False],
    [-0.3321733392943157, -4.025220792922246, False],
    [-0.3153488506350769, -4.095704751675652, False],
    [-0.315004115712085, -4.09714508228151, False],
    [-0.315004115712085, -4.09714508228151, False],
    [-0.3077690846507245, -4.159875471417938, False],
    [-0.3077690846507245, -4.159875471417938, False],
    [-0.29266213875731173, -4.221745725582512, False],
    [-0.28795150518370105, -4.243094224787444, False],
    [-0.28795150518370105, -4.243094224787444, True],
    [-0.28795150518370105, -4.243094224787444, True],
    [-0.2764134511660047, -4.322811474477992, True],
    [-0.2764134511660047, -4.322811474477992, True],
    [-0.27651249321375093, -4.325041909758191, True],
    [-0.2764301243873063, -4.3247813257817125, True],
    [-0.2764301243873063, -4.3247813257817125, True],
    [-0.2768523526602419, -4.322795857041512, True],
    [-0.2768523526602419, -4.322795857041512, True],
    [-0.27340857110966665, -4.347341880747472, True],
    [-0.27340857110966665, -4.347341880747472, True],
    [-0.27156380133034885, -4.402867566203427, False],
    [-0.27149116177084676, -4.407496048863387, False],
    [-0.27075350582216784, -4.4287272012141505, False],
    [-0.27075350582216784, -4.4287272012141505, False],
    [-0.2743262015601392, -4.490997014625869, False],
    [-0.27435794735002006, -4.492579778910196, False],
    [-0.27435794735002006, -4.492579778910196, False],
    [-0.27256725120130965, -4.512301061453465, False],
    [-0.27256725120130965, -4.512301061453465, False],
    [-0.2544025809381848, -4.536967880544419, False],
    [-0.2544025809381848, -4.536967880544419, False],
    [-0.22733976019126734, -4.575716719300206, False],
    [-0.22494804771377785, -4.578759593345965, False],
    [-0.22494804771377785, -4.578759593345965, False],
    [-0.20388993811158762, -4.613605070484205, False],
    [-0.20388993811158762, -4.613605070484205, False],
    [-0.18353791081515525, -4.652960723057323, False],
    [-0.18353791081515525, -4.652960723057323, False],
    [-0.16969983825523582, -4.681548133380017, False],
    [-0.16818293966215359, -4.6847399851620875, False],
    [-0.16818293966215359, -4.6847399851620875, False],
    [-0.160118416002213, -4.708013652190813, False],
    [-0.160118416002213, -4.708013652190813, False],
    [-0.15286039908753246, -4.72077843771262, False],
    [-0.15286051353319696, -4.720794788061025, False],
    [-0.15286051353319696, -4.720794788061025, False],
    [-0.15764400262857003, -4.716451955473296, True],
    [-0.15764400262857003, -4.716451955473296, True],
    [-0.1556671319741244, -4.7177410684737655, True],
    [-0.1556671319741244, -4.7177410684737655, False],
    [-0.15600282017849232, -4.71879881120324, False],
    [-0.15591751927552788, -4.718777057175428, False],
    [-0.15591751927552788, -4.718777057175428, False],
    [-0.1599365497383628, -4.726497426566214, False],
    [-0.1599365497383628, -4.726497426566214, False],
    [-0.1591038482654311, -4.728261592866537, False],
    [-0.1591038482654311, -4.728261592866537, False],
    [-0.16039051885546562, -4.727095514089071, False],
    [-0.1603802239555533, -4.727176730428451, False],
    [-0.1603802239555533, -4.727176730428451, False],
    [-0.15548298471735633, -4.72665915885103, False],
    [-0.14955054860177508, -4.72721972970447, False],
    [-0.14955054860177508, -4.72721972970447, False],
    [-0.14955054860177508, -4.72721972970447, False],
    [-0.09326418067377518, -4.732587204786049, False],
    [-0.09326418067377518, -4.732587204786049, False],
    [-0.09087410786769168, -4.737007695950639, False],
    [-0.09087410786769168, -4.737007695950639, False],
    [-0.08060867327483101, -4.7546566635066325, False],
    [-0.07930461318149151, -4.756541586714978, False],
    [-0.07930461318149151, -4.756541586714978, False],
    [-0.07229195377487663, -4.770751295900074, False],
    [-0.07229195377487663, -4.770751295900074, False],
    [-0.06399285516812936, -4.785625531595768, False],
    [-0.06399285516812936, -4.785625531595768, False],
    [-0.05677291038599208, -4.798707843436332, False],
    [-0.055816605736918934, -4.800694455906211, False],
    [-0.055816605736918934, -4.800694455906211, False],
    [-0.047964929241248844, -4.816022689101133, False],
    [-0.047964929241248844, -4.816022689101133, False],
    [-0.04016375317659948, -4.830571693052898, False],
    [-0.04003153531695769, -4.8308991360267655, False],
    [-0.04003153531695769, -4.8308991360267655, False],
    [-0.032247843117915176, -4.8483823402109385, False],
    [-0.032247843117915176, -4.8483823402109385, False],
    [-0.024250792384888364, -4.8560236322267905, False],
    [-0.024250792384888364, -4.8560236322267905, False],
    [-0.01628726781583667, -4.8578326287336155, False],
    [-0.015685813772698398, -4.8579972784984236, False],
    [-0.015685813772698398, -4.8579972784984236, False],
    [-0.005502770027491111, -4.861864966672483, False],
    [-0.0025728896993791883, -4.863258410146875, False],
    [-0.0006530348355441307, -4.868230979613023, False],
    [-0.0015902713203070906, -4.872818621993001, False],
    [-0.0029079035814298937, -4.881907421307841, False],
    [-0.002932292731337993, -4.883262261655876, False],
    [-0.002932292731337993, -4.883262261655876, True],
    [-0.0011110803468226127, -4.894648830594797, True],
    [-0.0009588222921910194, -4.897996444216859, True],
    [-0.0013824958098913598, -4.904907786168011, False],
    [-0.000651185694399703, -4.909147115674595, False],
    [0.00011207591086537239, -4.910984671656989, False],
    [0.001172868609843336, -4.913968294485954, False],
    [0.004431977181544644, -4.920990183339943, False],
    [0.005505823191344517, -4.9229903739006105, False],
    [0.006969767420146211, -4.9280400226494265, False],
    [0.006744008979296612, -4.927540643361569, False],
    [0.006814117989740853, -4.927550390710583, False],
    [0.007028911839472962, -4.927463288394096, False],
    [0.007139086606917311, -4.927401280466934, False],
    [0.008498141063949416, -4.9271223119578105, False],
    [0.019704654064453764, -4.92660281433761, True],
    [0.019704654064453764, -4.92660281433761, True],
    [0.03275116376748721, -4.926643440545843, True],
    [0.03204315444214201, -4.926598092662674, True],
    [0.026290955344286826, -4.92707934146832, True],
    [0.025803095687340627, -4.927067905112203, True],
    [0.026845960134329383, -4.927006103470224, True],
    [0.027532947511068162, -4.927212651214907, True],
    [0.027813562263784508, -4.927220213914992, True],
    [0.027609855786610682, -4.927126813997255, True],
    [0.0275909595161703, -4.9271101247606985, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262179, -4.9271133127634785, True],
    [0.02759457017262179, -4.9271133127634785, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
    [0.02759457017262168, -4.927113312763478, True],
]
VARIABLE_DECOMPOSITION = []


def app_init():
    APP.title = "Anomaly Detection on Kairos"
    APP.layout = html.Div(
        id="app",
        children=[
            # Global interval for refreshing data
            dcc.Interval(
                id="interval-component",
                interval=1000,
                n_intervals=0
            ),
            # Banner
            html.Div(
                id="banner",
                className="banner text-5xl",
                children=[html.Div(className="fa fa-chart-bar text-red-700"),
                          html.H3("Anomaly Detection", className="ml-2 text-gray-700")]
            ),
            # Left column
            html.Div(
                id="left-column",
                className="three columns",
                children=[
                ],
            ),
            # Right column
            html.Div(
                id="right-column",
                className="nine columns",
                children=[
                    html.Div(
                        id="charts",
                        className="flex bg-white mt-2",
                        children=[
                            html.Div(
                                className="p-8",
                                children=[
                                    html.P(
                                        className="text-center text-3xl border-b-[4px] border-orange-500",
                                        children="ICE Lab"
                                    ),
                                    dcc.Graph(
                                        id="ice_lab_chart", figure={},
                                        style={"height": "600px", "width": "400px"}
                                    ),
                                ]
                            ),
                            html.Div(
                                className="p-8 w-full",
                                children=[
                                    html.P(
                                        className="text-center text-3xl border-b-[4px] border-orange-500",
                                        children="ICE Lab"
                                    ),
                                    dcc.Graph(
                                        id="decomposition_chart", figure={},
                                        style={"height": "600px"}
                                    ),
                                ]
                            )

                        ]
                    ),

                ],
            ),
        ],
    )


@SERVER.route("/map_position_insert", methods=['GET', 'POST'])
def map_position_insert():
    global MAP_POSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    # bytes to dict
    data = json.loads(request.data.decode('utf-8'))
    MAP_POSITION.append([data["X"], data["Y"], data["anomaly"]])
    print(len(MAP_POSITION))
    # update_map_position()
    # Return 200 OK
    return "OK", 200


@SERVER.route("/variable_decomposition_insert", methods=['GET', 'POST'])
def variable_decomposition_insert():
    global VARIABLE_DECOMPOSITION
    if request.method != 'POST':
        return 'Method not allowed', 405
    data = json.loads(request.data.decode('utf-8'))
    VARIABLE_DECOMPOSITION.append(data)
    # Return 200
    return "OK", 200


@SERVER.route("/commit", methods=['GET', 'POST'])
def commit():
    global MAP_POSITION, VARIABLE_DECOMPOSITION
    if request.method == 'GET':
        # Here we reset the data
        MAP_POSITION, VARIABLE_DECOMPOSITION = [], []
        return "OK", 200
    # TODO: implement save to database
    return "Not implemented", 501


# --------
# Callbacks
# --------
import plotly.graph_objects as go


@callback(
    Output(component_id='ice_lab_chart', component_property='figure'),
    Input(component_id='interval-component', component_property='n_intervals')
)
def update_map_position(n_intervals):
    global MAP_POSITION
    print(str(len(MAP_POSITION)) + " chart")
    # fig_map = make_subplots(rows=1)
    fig_map = go.Figure()
    # plot map
    fig_map.add_layout_image(
        source="assets/ICE_lab.png",
        y=1,
        x=-1.5,
        sizex=12,
        sizey=15,
        xref="x",
        yref="y",
        opacity=1,
        layer="below",
        sizing="contain"
    )
    # set limits
    fig_map.update_layout(
        xaxis_range=[-1.5, 2.5],
        yaxis_range=[-5, 1],
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.4,
            bgcolor="rgba(255, 255, 255, 0.9)"

        )
    )
    # no point to display :(
    if len(MAP_POSITION) == 0:
        return fig_map
    # store our points
    points_copy = np.array(MAP_POSITION)
    # subtract offset to the Y coordinate to fit the image
    x_plot, y_plot = points_copy[1:, 0], points_copy[1:, 1]
    anomaly = points_copy[1:, 2]
    fig_map.add_traces([
        # correct behaviour
        go.Scatter(
            x=x_plot[anomaly == 0],
            y=y_plot[anomaly == 0],
            mode="markers",
            marker={"color": "blue"},
            name="Correct behaviour"
        ),
        # anomalies
        go.Scatter(
            x=x_plot[anomaly == 1],
            y=y_plot[anomaly == 1],
            mode="markers",
            marker={"color": "red"},
            name="Anomaly"
        ),
        # start and stop
        go.Scatter(
            x=[points_copy[0][0], points_copy[-1][0]], y=[points_copy[0][1], points_copy[-1][1]],
            mode="markers",
            marker={"color": "orange", "size": 12},
            name="Start and end",
            showlegend=False
        )
    ])
    # map.legend(handles=[green_patch, blue_patch, red_patch], loc=1)

    return fig_map


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8050
    debug = True
    # Initialize the app
    app_init()
    # Start the app
    APP.run(debug=debug, host=host, port=port)
