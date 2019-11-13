# Notes for LASSI

## Connecting to Windows host

To get into windows PC from linux:

```
xfreerdp --no-nla --plugin cliprdr -g 85% lassi
```

## LASSI Manager sequence view

```
@startuml
turtle -> manager: begin
manager -> DAQ: run measurement
DAQ -> TLS: deploy
DAQ -> TLS: power up
TLS -> DAQ: ready
DAQ -> TLS: configure
DAQ -> TLS: start measurement
TLS -> DAQ: measurement complete
TLS -> DAQ : raw data
DAQ -> TLS : power down
DAQ -> manager : measurement complete

DAQ -> PP: raw data
manager -> PP: start PP command
DAQ ---> manager: raw data

manager ---> NAS : raw data (FITS file)

manager -> turtle: complete (reference meas. only)
PP -> NAS : processed data (FITS file)
PP -> manager: PP done Zs (signal meas. only)
PP -> NAS : Zernike data (FITS file)
manager -> "active surface": update Zs (signal meas. only)
manager -> turtle: complete
@enduml
```

## LASSI DAQ state diagram

```
@startuml
Ready : The scanner is powered up and ready for a command
Paused : The scan was running, but has been paused
Calculating : Not sure. Computing a result?
Imaging : Acquiring images (i.e. pictures, not ranges)
CalibratingTiltSensor : I expect this means what it sounds like.
CalibratingTiltSensor : As a note, the tilt sensor cannot be used
CalibratingTiltSensor : in the inverted mounting position.
Scanning : Scanner is acquiring range data
PreparingScan : Scan command has been given, but scanner
PreparingScan : is performing internal preparations to scan,
PreparingScan : like positioning mount or spinning up the mirror.
Initializing : TBD
Running : TBD
Failure : Scanner has failed to complete an operation

[*] --> Ready
Ready --> StartScan
StartScan --> PreparingScan
PreparingScan --> ScannerReady
ScannerReady --> Scanning
Scanning : Scanner is acquiring range data
Scanning --> ScanComplete
ScanComplete --> Ready

Ready --> CalibratingTiltSensor
CalibratingTiltSensor --> Ready

StartScan --> Failure
PreparingScan --> Failure
ScannerReady --> Failure
Scanning --> Failure

StartScan --> Paused
PreparingScan --> Paused
ScannerReady --> Paused
Scanning --> Paused
@enduml
```

## LASSI state diagram

```
@startuml
[*] --> Ready
Ready --> TLS_measurement : start
TLS_measurement : measure GBT primary
TLS_measurement : surface
TLS_measurement --> Stop_TLS : done
Stop_TLS : clean shutdown of
Stop_TLS : hardware
Stop_TLS --> Ready : abort event
Stop_TLS --> Post_Processing : data
Post_Processing --> Shutdown_Post_Processing : abort event
Post_Processing --> Shutdown_Post_Processing : done
Post_Processing --> Comparison : comparison measurement
Comparison : compare AOCS with current data
Post_Processing --> Ready : AOCS measurement
Comparison --> Shutdown_Post_Processing : abort event
Shutdown_Post_Processing : clean shutdown of PP module
Shutdown_Post_Processing --> Ready : abort event
@enduml
```

## LASSI decomposition context diagram
```
@startuml
package "GBT M&C" {
  package ScanCoordinator
  package ActiveSurface
  package RcvrArray75_115
  package Grail
  package "Other components..."
  package LASSI #lightblue
}
@enduml
```

## LASSI module uses context diagram
```
@startuml
package "GBT M&C" {
package LASSI #lightblue {
  package Manager
  package TLS_Scanner
  package LASSI_DAQ
  package LASSI_post_processing
}
package turtle_server
package Active_Surface

turtle_server --> LASSI : <<use>>
LASSI --> Active_Surface : <<use>>
}
@enduml
```

## LASSI C&C view
```
@startuml
skinparam linetype polyline

package "turtle server" as turtle

package "LASSI" {
  [LASSI Manager] as manager
  [LASSI DAQ] as DAQ
  [scanner enclosure] as enclosure
  [TLS scanner] as scanner
  [LASSI Post-Processing] as PP

  DAQ-->PP : scan data
  DAQ-->scanner : commands
  scanner-->DAQ : point cloud

  DAQ-->enclosure : commands

  DAQ-->PP : "scan data"

  manager<-left->DAQ : C&C
  DAQ-left->manager : "scan data"
  manager<--->PP : C&C
  PP-right->manager : Zernike
}

package "active surface" as AS

turtle-->LASSI : "scan request"
LASSI-->AS : commands
@enduml
```

## LASSI DAQ context
```
@startuml
[LASSI DAQ] #lightblue
[LASSI DAQ] -up- [LASSI Manager] : <<use>>
[TLS Scanner] -right- [LASSI DAQ] : <<use>>
[LASSI DAQ] -right- [LASSI Post-Processing] : <<use>>
@enduml
```

## LASSI PP context
```
@startuml
[LASSI Post-Processing] as PP #lightblue
[LASSI DAQ] <.right. PP : <<use>>
PP <.right.> [manager] : <<use>>
@enduml
```

## LASSI interoperability sequence
```
@startuml
user -> turtle : AutoLASSI()
turtle -> manager : prepare
turtle --> antenna : slew
manager -> LASSI : power up and deploy TLS
LASSI -> manager : ready
manager -> turtle : ready
antenna -> turtle : ready
turtle -> manager : do LASSI measurement
manager -> LASSI : do LASSI measurement
LASSI -> manager : TLS done
manager -> turtle : done
manager -> LASSI : power down and retract TLS
turtle --> antenna : slew
LASSI -> LASSI : data reduction
LASSI -> manager : Zernikes
manager -> turtle : Zernikes
turtle -> antenna : active surface corrections
turtle -> user : done
@enduml
```

# LASSI interoperability context
```
@startuml
skinparam linetype polyline

package "GBT M&C" {
  package "Antenna manager" as Antenna
  package turtle as "turtle server"
  package "LASSI manager" as manager
  package LASSI #lightblue {
    node "LASSI DAQ"
    node "LASSI PP"
    node "TLS Scanner"
  }

  manager --> LASSI : <<use>>
  turtle --> Antenna : <<use>>
  turtle --> manager : <<use>>

}
@enduml
```
