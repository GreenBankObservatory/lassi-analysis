    
AutoLassi sequence:   
   
    sequenceDiagram
    participant AL as AutoLassi
    participant US as user.Scan
    participant UT as user.Telescope
    participant YT as ygor.Telescope
    participant YA as ygor.ActiveSurface
    participant AC as ActSurfaceConnection
    participant Q as Queue
    participant LP as gfm.LassiPlugin
    participant TC as TurtleClient
    participant TS as TurtleServer
    AL ->> AL: Scan -> Execute
    AL ->> US: Execute -> Stop
    Note left of US: LASSI created Zs
    Note left of US: see other diagram
    AL ->> US: Execute -> UpdateZs
    US ->> UT: UpdateZs -> UpdateZs
    UT ->> YT: UpdateZs -> UpdateZs
    YT ->> YA: UpdateZs -> UpdateZs
    YA ->> Q: UpdateZs -> get
    Note left of Q: block here 
    LP ->> LP: OnEndScan -> UpdateZs
    Note left of LP: picks up Zs
    Note left of LP: user interaction here
    LP ->> TC: UpdateZs -> UpdateZs
    TC ->> TS: UpdateZs -> UpdateZs
    TS ->> YT: UpdateZs -> PutZs
    YT ->> YA: PutZs -> PutZs
    YA ->> Q: PutZs -> put
    Note left of Q: unblocks
    YA ->> AC: UpdateZs -> SetZs
    Note left of AC: now free to send Zs
   

Lassi Manager sequence:

sequenceDiagram
participant LM as LassiManager
participant TC as TLSClient
participant FW as FITSWriter
participant LD as lassi_daq
participant AC as AnalysisClient
participant LA as LassiAnalysis
participant G as GPUs
LM ->> TC: doStart -> powerUp
TC ->> LD: powerUp -> powerUp
LM ->> TC: doStart -> configure
TC ->> LD: configure -> configure
LM ->> TC: doArm -> startScan
LM ->> FW: doArm -> startScan
TC ->> LD: startScan -> startScan
LD ->> TC: scanDone -> scanDone
TC ->> LM: scanDone -> TlsScanDone
LM ->> FW: TlsScanDone -> scanDone
Note right of FW: Raw FITS data written
LM ->> AC: TlsScanDone -> startProcessing
AC ->> LA: startProcessing -> startProcessing
LA ->> G: startProcessing -> smooth
G ->> LA: smooth -> smoothDone
LA ->> LA: writeResults
Note left of LA: Processed FITS written
LA ->> AC: doneProcessing -> doneProcessing
AC ->> LM: doneProcessing
LM ->> LM: doStop
