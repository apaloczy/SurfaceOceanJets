maxangr_thresh = 75     # Maximum angle to perform rotation.
maxangmdt_thresh = None # Used only to find the best ground track.
pgud_thresh = None
maxd0_thresh = None
flagl, flagr = None, None
ADT_thresh = None
us_thresh = None
Ro_thresh = None

if name=="GulfStream33N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = -76.65, 33, 5, "Gulf Stream 33N", "Gulf-Stream-33N_xstream_transect_DTU15.npz", 0, 0

if name=="GulfStream34N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = -75.1, 34.5, 5, "Gulf Stream 34N", "Gulf-Stream-33N_xstream_transect_DTU15.npz", 0, 0

if name=="GulfStream36N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = -73.3, 36.5, 10, "Gulf Stream 36N", "Gulf-Stream_xstream_transect_DTU15.npz", 0, 0

if name=="GulfStream37N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = -71.2, 37.2, 10, "Gulf Stream 37N", "Gulf-Stream_xstream_transect_DTU15.npz", 0, 0

if name=="GulfStream38N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = -69, 38, 10, "Gulf Stream 38N", "Gulf-Stream_xstream_transect_DTU15.npz", 0, 0

if name=="KuroshioCurrent25N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = 122.7, 25, 5, "Kuroshio Current 25N", "Kuroshio-Current-25N_xstream_transect_DTU15.npz", 0, 0
    flagl = 2

if name=="KuroshioCurrent28p5N":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = 127.23, 28.5, 5, "Kuroshio Current 28p5N", "Kuroshio-Current-28p5N_xstream_transect_DTU15.npz", 0, 0
    flagl = 2

if name=="AgulhasCurrent":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = 28, -34, 4, "Agulhas Current", "Agulhas-Current_xstream_transect_DTU15.npz", 0, 0

if name=="EAC29S":
    wanted_lon, wanted_lat, d, sup, fname_xstream, xnudge, ynudge = 154, -29, 5, "East Australian Current 29S", "East-Australian-Current-29S_xstream_transect_DTU15.npz", 0, 1.1

if name=="BrazilCurrent29S":
    wanted_lon, wanted_lat, sup, d, fname_xstream, xnudge, ynudge = -47.3, -29, "Brazil Current 29S", 5, "Brazil-Current-29S_xstream_transect_DTU15.npz", 0.25, -0.5

if pgud_thresh is None:
    pgud_thresh = 50

if ADT_thresh is None:
    ADT_thresh = 5 # [m]

if us_thresh is None:
    us_thresh = 3 # [m/s]

if Ro_thresh is None:
    Ro_thresh = 2.0

if maxd0_thresh is None:
    maxd0_thresh = 70 # [km]

if flagl is None:
    flagl = 1

if flagr is None:
    flagr = 3071

if maxangmdt_thresh is None:
    maxangmdt_thresh = 75
