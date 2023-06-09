import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(1, 5),
    cirq.GridQubit(1, 6),
    cirq.GridQubit(1, 7),
    cirq.GridQubit(2, 4),
    cirq.GridQubit(2, 5),
    cirq.GridQubit(2, 6),
    cirq.GridQubit(2, 7),
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(3, 7),
    cirq.GridQubit(4, 2),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(4, 7),
    cirq.GridQubit(5, 2),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
    cirq.GridQubit(6, 2),
    cirq.GridQubit(6, 3),
    cirq.GridQubit(6, 4),
    cirq.GridQubit(6, 5),
    cirq.GridQubit(7, 2),
    cirq.GridQubit(7, 3),
    cirq.GridQubit(7, 4),
]

CIRCUIT = cirq.Circuit(
    moments=[
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 7)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 2)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -2.079870303178702).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 2.0436918407499873).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 1.2371391697444234).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -1.2825274365288457).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -0.6529975013575373).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 0.21248377848559125).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 0.2767373377033284).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -0.18492941569567625).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 0.02232591119805812).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -0.030028573876142287).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -0.8467509808142173).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 0.8164932597686655).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -1.00125113388313).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 1.1224546746752684).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -0.16310561378711827).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 0.1766183348870303).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -0.22542387771877406).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 0.2814659583608806).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -0.33113463396189063).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 0.40440704518468423).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -0.4081262439699967).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 0.3666829187201306).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -0.3507308388473503).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 0.37554649493270875).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * -1.4187954353764791).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 1.5102819373895253).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5033136051987404, phi=0.5501439149572028).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
                ),
                cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.59182423935832, phi=-5.773664463980115).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
                ),
                cirq.FSimGate(theta=1.5516764540193888, phi=0.505545707839895).on(
                    cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 1.3803105504474993).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -1.4164890128762133).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -0.7660705551087533).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 0.7206822883243308).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 1.3183560383893944).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -1.7588697612613406).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -0.6722145774944012).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 0.7640224995020534).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 0.5799079899133832).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -0.5876106525914674).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 1.0843371101222938).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -1.1145948311678457).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 0.7990757781248072).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -0.6778722373326689).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -1.6258237067659351).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 1.6393364278658469).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 0.7948295009385445).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -0.7387874202964381).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 0.049341949396894985).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 0.02393046182589869).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 0.4710627118441926).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -0.5125060370940587).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 2.1645856475342256).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -2.1397699914488673).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * 1.2773117920270392).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -1.1858252900139932).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.X**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -5.435868884042397).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 5.438497289344933).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -5.19048555249959).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 5.170988862096221).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 2.5333591271878086).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -2.4748096263683066).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -4.480708067260001).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 4.525888267898699).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 2.135954522972214).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -2.1822665205802965).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -3.7780476633662574).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 3.817335880513747).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 0.7811374803446167).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -0.6780279413275597).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 1.863573798571082).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -2.150412392135508).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 2.3134893226730737).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -2.238493420699622).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 1.42630741834175).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -1.5270341780432073).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
                ),
                cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                    cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 5.79385605258612).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -5.791227647283584).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 5.223139057027918).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -5.242635747431287).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -2.346072351850546).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 2.404621852670048).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 5.048199817882042).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -5.0030196172433445).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -2.6543362735839113).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 2.6080242759758283).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 3.9045088495271663).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -3.8652206323796765).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -1.5516585295358842).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 1.6547680685529413).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -1.8933072151541963).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 1.6064686215897703).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -2.3490397609251703).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 2.424035662898622).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -1.8655832225378013).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 1.7648564628363437).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -6.214223110662173).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 6.24431588336413).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -6.196295096608877).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 6.191833422443152).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -5.367868774756692).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 5.257156584109544).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -1.6118072404137829).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 1.5665192386902935).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -5.408932498710608).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 5.396221422935972).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -3.2786928385561493).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 3.339006443218924).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -5.390755870544794).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 5.4172568990486605).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -5.620144773112766).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 5.630469153514815).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 4.367652291347506).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -3.9105776028384707).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 7.0181466269225865).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -7.000766026200176).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 5.700873278515409).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -5.683378195921049).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 4.586335789661189).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -4.76537552715921).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                    cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
                ),
                cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.596346344028619, phi=0.5104319949477776).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
                ),
                cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
                ),
                cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 6.89944406229822).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -6.869351289596263).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 6.506615138479995).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -6.511076812645719).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 6.150506057270183).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -6.2612182479173315).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 2.4087294851133443).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -2.4540174868368334).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 4.737705877923889).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -4.750416953698525).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 2.9425087256630427).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -2.882195121000268).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 4.466531408750767).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -4.440030380246901).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 4.486471496440378).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -4.476147116038329).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -4.89701654221443).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 5.354091230723465).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -5.629287261948809).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 5.646667862671219).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -5.760627714067928).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 5.778122796662288).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -3.985782702743221).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 3.806742965245199).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.X**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 2)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 2)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -2.4865845873665364).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 2.4890814068883764).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -2.4240781150731663).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 2.419398026235366).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 2.3861256785493166).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -2.392456163642626).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 12.703597923836748).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -12.7869629079138).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 12.184253063938954).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -12.108584830758572).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 3.782562501914174).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -3.873596611893716).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 4.772639843256901).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -4.771314675186062).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 8.49593730829863).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -8.479908941862229).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 9.60223181672896).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -9.605639326034064).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 6.330499004273446).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -6.2177071019033425).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 9.851852381617888).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -9.926465199012979).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 6.431104618355057).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -6.38660616379351).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.5040835324508275, phi=0.6761565725975858).on(
                    cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                    cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
                ),
                cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 2.557874433792943).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -2.555377614271102).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 1.9789952328325573).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -1.9836753216703575).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -2.805807436079691).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 2.7994769509863815).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -12.477250219528523).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 12.39388523545147).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -11.31088974563283).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 11.386557978813212).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -5.4898636407973544).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 5.398829530817813).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -5.863871460773714).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 5.8651966288445525).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -8.850693052252502).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 8.866721418688904).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -10.03456101076628).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 10.031153501461176).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -5.434421382024706).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 5.54721328439481).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -9.17988634353845).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 9.10527352614336).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -6.5670035038476025).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 6.61150195840915).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.Y**0.5).on(cirq.GridQubit(1, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -13.031870303178678).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 12.995691840749963).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 5.381139169744492).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -5.426527436528915).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -6.86899750135751).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 6.428483778485565).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 5.16073733770325).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -5.068929415695599).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -0.7176740888019262).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 0.7099714261238419).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -4.694750980814187).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 4.664493259768636).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -4.701251133883051).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 4.82245467467519).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 3.5368943862129347).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -3.523381665113022).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -1.113423877718808).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 1.1694659583609144).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -3.587134633961795).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 3.6604070451845887).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -5.2921262439699195).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 5.250682918720053).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -6.349327548997941).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 6.3741432050833).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * -7.486795435376533).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 7.578281937389579).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5033136051987404, phi=0.5501439149572028).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
                ),
                cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.59182423935832, phi=-5.773664463980115).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
                ),
                cirq.FSimGate(theta=1.5516764540193888, phi=0.505545707839895).on(
                    cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 12.332310550447476).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -12.36848901287619).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -4.910070555108823).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 4.864682288324399).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 7.534356038389369).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -7.974869761261314).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -5.556214577494324).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 5.648022499501975).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 1.3199079899133674).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -1.3276106525914517).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 4.932337110122265).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -4.9625948311678165).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 4.499075778124728).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -4.37787223733259).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -5.325823706765988).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 5.3393364278658995).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 1.682829500938578).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -1.6267874202964716).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 3.305341949396799).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -3.232069538174005).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 5.3550627118441145).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -5.39650603709398).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 8.163182357684818).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -8.138366701599459).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * 7.345311792027093).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -7.253825290014047).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -17.867868884042345).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 17.87049728934488).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -17.622485552499665).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 17.602988862096296).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 7.565359127187911).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -7.506809626368408).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -15.28470806725993).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 15.329888267898626).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 7.019954522972137).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -7.066266520580219).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -13.842047663366333).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 13.881335880513822).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 3.001137480344569).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -2.8980279413275123).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 5.563573798571002).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -5.8504123921354285).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 7.868086032823645).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -7.793090130850194).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 4.3863074183418185).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -4.487034178043276).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
                ),
                cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                    cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 18.225856052586064).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -18.223227647283533).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 17.655139057028).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -17.674635747431363).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -7.378072351850649).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 7.436621852670151).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 15.852199817881967).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -15.80701961724327).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -7.538336273583833).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 7.492024275975751).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 13.968508849527241).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -13.929220632379753).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -3.771658529535837).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 3.874768068552894).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -5.593307215154117).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 5.30646862158969).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -7.9036364710757425).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 7.978632373049194).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -4.825583222537869).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 4.724856462836412).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.Y**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -16.574223110662086).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 16.60431588336404).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -15.816295096608934).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 15.811833422443211).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -13.3598687747566).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 13.249156584109453).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -4.127807240413703).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 4.082519238690215).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -13.252932498710596).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 13.24022142293596).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -8.162692838556204).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 8.223006443218978).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -12.938755870544817).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 12.965256899048683).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -12.724144773112773).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 12.73446915351482).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 11.027652291347495).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -10.570577602838458).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 17.082146626922658).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -17.06476602620025).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 14.58087327851535).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -14.563378195920992).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 10.871739079510629).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -11.050778817008649).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                    cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
                ),
                cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.596346344028619, phi=0.5104319949477776).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
                ),
                cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
                ),
                cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 17.259444062298133).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -17.229351289596174).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 16.126615138480055).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -16.131076812645777).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 14.142506057270092).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -14.253218247917241).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 4.924729485113265).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -4.9700174868367535).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 12.581705877923879).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -12.594416953698515).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 7.826508725663096).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -7.7661951210003215).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 12.014531408750791).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -11.988030380246926).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 11.590471496440383).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -11.580147116038336).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -11.55701654221442).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 12.014091230723457).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -15.693287261948884).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 15.710667862671292).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -14.640627714067872).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 14.658122796662232).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -10.271185992592658).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 10.092146255094638).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 6)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -4.706584587366488).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 4.709081406888329).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -4.644078115073251).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 4.639398026235451).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 4.902125678549236).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -4.908456163642546).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 26.023597923836856).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -26.106962907913907).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 25.356253063938887).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -25.2805848307585).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 8.370562501914259).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -8.461596611893802).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 10.100639843256841).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -10.099314675186001).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 18.263937308298605).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -18.247908941862203).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 20.40623181672889).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -20.409639326033993).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 13.138499004273484).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -13.02570710190338).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 19.994449091768548).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -20.069061909163636).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 13.831104618355031).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -13.786606163793484).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.5040835324508275, phi=0.6761565725975858).on(
                    cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                    cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
                ),
                cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 4.777874433792896).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -4.775377614271054).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 4.198995232832642).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -4.203675321670441).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -5.321807436079611).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 5.315476950986302).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -25.79725021952863).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 25.713885235451578).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -24.48288974563276).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 24.55855797881315).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -10.07786364079744).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 9.986829530817898).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -11.191871460773655).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 11.193196628844492).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -18.61869305225248).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 18.63472141868888).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -20.83856101076621).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 20.835153501461107).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -12.242421382024746).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 12.35521328439485).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -19.32248305368911).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 19.24787023629402).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -13.967003503847575).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 14.01150195840912).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.Y**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 7)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -23.983870303178655).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 23.947691840749943).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 9.52513916974456).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -9.570527436528984).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -13.084997501357485).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 12.644483778485537).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 10.044737337703173).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -9.952929415695523).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -1.4576740888019104).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 1.4499714261238263).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -8.542750980814159).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 8.512493259768608).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -8.401251133882973).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 8.52245467467511).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 7.236894386212986).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -7.223381665113074).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -2.0014238777188416).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 2.057465958360948).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -6.843134633961698).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 6.916407045184491).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -10.176126243969842).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 10.134682918719976).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -12.347924259148533).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 12.372739915233888).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * -13.554795435376587).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 13.646281937389634).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5033136051987404, phi=0.5501439149572028).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
                ),
                cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.59182423935832, phi=-5.773664463980115).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
                ),
                cirq.FSimGate(theta=1.5516764540193888, phi=0.505545707839895).on(
                    cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 23.28431055044745).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -23.320489012876163).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -9.054070555108892).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 9.008682288324469).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 13.750356038389338).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -14.190869761261286).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -10.440214577494247).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 10.5320224995019).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 2.0599079899133517).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -2.067610652591436).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 8.780337110122234).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -8.810594831167785).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 8.199075778124648).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -8.07787223733251).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -9.025823706766039).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 9.039336427865951).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 2.570829500938612).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -2.5147874202965053).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 6.561341949396702).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -6.48806953817391).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 10.239062711844038).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -10.280506037093904).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 14.161779067835406).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -14.136963411750049).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * 13.413311792027148).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -13.3218252900141).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.X**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 7)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -30.29986888404229).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 30.302497289344824).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -30.054485552499738).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 30.034988862096366).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 12.597359127188014).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -12.538809626368511).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -26.08870806725985).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 26.13388826789855).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 11.90395452297206).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -11.950266520580142).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -23.906047663366408).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 23.945335880513902).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 5.221137480344522).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -5.118027941327464).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 9.263573798570924).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -9.55041239213535).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 13.422682742974219).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -13.34768684100077).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 7.346307418341885).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -7.447034178043343).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
                ),
                cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                    cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 30.657856052586013).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -30.65522764728348).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 30.087139057028068).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -30.106635747431437).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -12.410072351850753).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 12.468621852670255).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 26.656199817881895).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -26.611019617243198).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -12.422336273583753).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 12.376024275975672).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 24.032508849527318).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -23.993220632379824).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -5.991658529535789).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 6.094768068552847).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -9.293307215154037).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 9.006468621589612).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -13.45823318122632).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 13.53322908319977).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -7.785583222537938).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 7.68485646283648).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 3)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 2)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -26.934223110661993).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 26.964315883363945).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -25.436295096608994).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 25.43183342244327).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -21.351868774756507).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 21.24115658410936).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -6.643807240413623).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 6.598519238690134).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -21.096932498710586).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 21.084221422935954).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -13.046692838556257).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 13.107006443219033).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -20.486755870544844).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 20.51325689904871).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -19.82814477311278).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 19.838469153514826).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 17.687652291347487).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -17.230577602838448).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 27.146146626922736).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -27.128766026200324).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 23.46087327851529).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -23.443378195920936).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 17.157142369360066).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -17.33618210685809).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                    cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
                ),
                cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.596346344028619, phi=0.5104319949477776).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
                ),
                cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
                ),
                cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 27.61944406229804).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -27.589351289596088).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 25.746615138480117).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -25.75107681264584).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 22.13450605727).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -22.245218247917148).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 7.440729485113184).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -7.486017486836674).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 20.425705877923868).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -20.4384169536985).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 12.71050872566315).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -12.650195121000372).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 19.562531408750814).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -19.53603038024695).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 18.69447149644039).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -18.684147116038343).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -18.21701654221441).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 18.674091230723448).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -25.757287261948953).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 25.774667862671368).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -23.52062771406781).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 23.538122796662165).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -16.556589282442097).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 16.377549544944078).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.X**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -6.926584587366442).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 6.929081406888282).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -6.864078115073335).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 6.859398026235534).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 7.418125678549155).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -7.424456163642465).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 39.34359792383697).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -39.42696290791402).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 38.52825306393881).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -38.452584830758425).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 12.958562501914345).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -13.049596611893888).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 15.428639843256777).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -15.42731467518594).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 28.031937308298577).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -28.01590894186218).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 31.210231816728815).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -31.213639326033913).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 19.946499004273523).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -19.833707101903418).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 30.137045801919207).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -30.211658619314296).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 21.231104618355).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -21.186606163793456).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.5040835324508275, phi=0.6761565725975858).on(
                    cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                    cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
                ),
                cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 6.997874433792849).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -6.995377614271008).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 6.418995232832726).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -6.423675321670527).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -7.8378074360795305).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 7.831476950986221).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -39.11725021952874).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 39.03388523545169).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -37.65488974563269).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 37.730557978813074).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -14.665863640797525).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 14.574829530817984).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -16.519871460773594).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 16.52119662884443).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -28.386693052252454).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 28.402721418688852).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -31.64256101076613).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 31.63915350146103).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -19.050421382024783).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 19.16321328439489).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -29.465079763839764).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 29.390466946444676).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -21.367003503847553).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 21.411501958409097).on(cirq.GridQubit(6, 5)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 7)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(2, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(4, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -34.93587030317863).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 34.899691840749924).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * 13.66913916974463).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * -13.714527436529053).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -19.300997501357458).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 18.86048377848551).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * 14.928737337703097).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -14.836929415695444).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -2.1976740888018944).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 2.1899714261238103).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * -12.39075098081413).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * 12.360493259768578).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -12.10125113388289).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 12.22245467467503).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 10.936894386213037).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -10.923381665113125).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * -2.8894238777188748).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * 2.945465958360982).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -10.099134633961603).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 10.172407045184396).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -15.060126243969762).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * 15.018682918719897).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -18.34652096929912).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 18.371336625384476).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * -19.622795435376638).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * 19.714281937389686).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5033136051987404, phi=0.5501439149572028).on(
                    cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
                ),
                cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                    cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
                ),
                cirq.FSimGate(theta=1.59182423935832, phi=-5.773664463980115).on(
                    cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
                ),
                cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                    cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
                ),
                cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                    cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
                ),
                cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                    cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
                ),
                cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                    cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
                ),
                cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                    cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
                ),
                cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                    cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
                ),
                cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                    cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
                ),
                cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                    cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
                ),
                cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                    cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
                ),
                cirq.FSimGate(theta=1.5516764540193888, phi=0.505545707839895).on(
                    cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 34.236310550447435).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -34.27248901287614).on(cirq.GridQubit(1, 7)),
                cirq.Rz(np.pi * -13.19807055510896).on(cirq.GridQubit(2, 4)),
                cirq.Rz(np.pi * 13.152682288324536).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 19.96635603838931).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -20.40686976126126).on(cirq.GridQubit(2, 7)),
                cirq.Rz(np.pi * -15.32421457749417).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 15.416022499501823).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 2.7999079899133363).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -2.80761065259142).on(cirq.GridQubit(3, 7)),
                cirq.Rz(np.pi * 12.628337110122207).on(cirq.GridQubit(4, 2)),
                cirq.Rz(np.pi * -12.658594831167758).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 11.899075778124569).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -11.777872237332431).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -12.725823706766091).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 12.739336427866004).on(cirq.GridQubit(4, 7)),
                cirq.Rz(np.pi * 3.458829500938646).on(cirq.GridQubit(5, 2)),
                cirq.Rz(np.pi * -3.4027874202965385).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 9.817341949396608).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -9.744069538173814).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 15.12306271184396).on(cirq.GridQubit(6, 2)),
                cirq.Rz(np.pi * -15.164506037093826).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 20.160375777985994).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -20.13556012190064).on(cirq.GridQubit(6, 5)),
                cirq.Rz(np.pi * 19.481311792027203).on(cirq.GridQubit(7, 2)),
                cirq.Rz(np.pi * -19.389825290014155).on(cirq.GridQubit(7, 3)),
            ]
        ),
        cirq.Moment(
            operations=[
                (cirq.Y**0.5).on(cirq.GridQubit(1, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 6)),
                (cirq.X**0.5).on(cirq.GridQubit(1, 7)),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(2, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 6)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 7)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 5)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 7)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(5, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(6, 3)),
                (cirq.X**0.5).on(cirq.GridQubit(6, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * -42.731868884042235).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * 42.73449728934477).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * -42.48648555249982).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * 42.46698886209646).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * 17.629359127188117).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * -17.570809626368614).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * -36.89270806725978).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * 36.93788826789848).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * 16.787954522971983).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * -16.834266520580062).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * -33.970047663366486).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * 34.00933588051398).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * 7.441137480344476).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * -7.338027941327417).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * 12.963573798570843).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * -13.250412392135269).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * 18.97727945312479).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * -18.902283551151342).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * 10.306307418341955).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * -10.407034178043412).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                    cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
                ),
                cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                    cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
                ),
                cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                    cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
                ),
                cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                    cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
                ),
                cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                    cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
                ),
                cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                    cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
                ),
                cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                    cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
                ),
                cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                    cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
                ),
                cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                    cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
                ),
                cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                    cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
                ),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.Rz(np.pi * 43.08985605258596).on(cirq.GridQubit(1, 5)),
                cirq.Rz(np.pi * -43.08722764728342).on(cirq.GridQubit(1, 6)),
                cirq.Rz(np.pi * 42.51913905702814).on(cirq.GridQubit(2, 5)),
                cirq.Rz(np.pi * -42.53863574743151).on(cirq.GridQubit(2, 6)),
                cirq.Rz(np.pi * -17.442072351850854).on(cirq.GridQubit(3, 3)),
                cirq.Rz(np.pi * 17.500621852670356).on(cirq.GridQubit(3, 4)),
                cirq.Rz(np.pi * 37.46019981788182).on(cirq.GridQubit(3, 5)),
                cirq.Rz(np.pi * -37.415019617243125).on(cirq.GridQubit(3, 6)),
                cirq.Rz(np.pi * -17.306336273583675).on(cirq.GridQubit(4, 3)),
                cirq.Rz(np.pi * 17.260024275975592).on(cirq.GridQubit(4, 4)),
                cirq.Rz(np.pi * 34.09650884952739).on(cirq.GridQubit(4, 5)),
                cirq.Rz(np.pi * -34.057220632379895).on(cirq.GridQubit(4, 6)),
                cirq.Rz(np.pi * -8.211658529535743).on(cirq.GridQubit(5, 3)),
                cirq.Rz(np.pi * 8.3147680685528).on(cirq.GridQubit(5, 4)),
                cirq.Rz(np.pi * -12.993307215153958).on(cirq.GridQubit(5, 5)),
                cirq.Rz(np.pi * 12.706468621589535).on(cirq.GridQubit(5, 6)),
                cirq.Rz(np.pi * -19.012829891376892).on(cirq.GridQubit(6, 3)),
                cirq.Rz(np.pi * 19.08782579335034).on(cirq.GridQubit(6, 4)),
                cirq.Rz(np.pi * -10.745583222538006).on(cirq.GridQubit(7, 3)),
                cirq.Rz(np.pi * 10.644856462836547).on(cirq.GridQubit(7, 4)),
            ]
        ),
        cirq.Moment(
            operations=[
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 5)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(1, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(1, 7)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 5)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(2, 6)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(2, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(3, 3)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(3, 4)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 5)),
                (cirq.X**0.5).on(cirq.GridQubit(3, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(3, 7)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 2)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 4)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(4, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(4, 6)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(4, 7)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 2)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 3)),
                (cirq.Y**0.5).on(cirq.GridQubit(5, 4)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(5, 5)
                ),
                (cirq.X**0.5).on(cirq.GridQubit(5, 6)),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 2)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 3)
                ),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(6, 4)
                ),
                (cirq.Y**0.5).on(cirq.GridQubit(6, 5)),
                (cirq.Y**0.5).on(cirq.GridQubit(7, 2)),
                (cirq.X**0.5).on(cirq.GridQubit(7, 3)),
                cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                    cirq.GridQubit(7, 4)
                ),
            ]
        ),
    ]
)
