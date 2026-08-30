"""Generate sample STEP files to exercise the pipeline."""
import os
import cadquery as cq

out = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out, exist_ok=True)

def save(wp, name):
    cq.exporters.export(wp, os.path.join(out, name + ".step"))
    print("wrote", name)

# 1. sheet-metal-ish bracket plate with a bolt pattern
plate = (cq.Workplane("XY").box(160, 90, 6)
         .edges("|Z").fillet(10)
         .faces(">Z").workplane()
         .pushPoints([(-60, -30), (-60, 30), (60, -30), (60, 30)]).hole(9)
         .faces(">Z").workplane().hole(40))
save(plate, "bracket_plate")

# 2. machined block: counterbores, blind hole, pocket
blk = (cq.Workplane("XY").box(90, 60, 30)
       .edges("|Z").fillet(6)
       .faces(">Z").workplane()
       .pushPoints([(-32, -20), (32, -20), (-32, 20), (32, 20)])
       .cboreHole(6.6, 11, 6.5)
       .faces(">Z").workplane().hole(25, depth=18)
       .faces(">X").workplane(centerOption="CenterOfBoundBox").hole(8))
save(blk, "machined_block")

# 3. flange with bolt circle + countersinks
fl = (cq.Workplane("XY").circle(70).extrude(12)
      .faces(">Z").workplane().hole(45)
      .faces(">Z").workplane().polarArray(55, 0, 360, 6).cskHole(9, 18, 82))
save(fl, "flange")

# 4. laser-cut sheet part
sheet = (cq.Workplane("XY").rect(220, 120).extrude(3)
         .edges("|Z").fillet(8)
         .faces(">Z").workplane()
         .rarray(40, 40, 5, 3).hole(6.5))
save(sheet, "laser_panel")

# 5. a five-item assembly, so the pipeline has a part with real sub-parts:
#    body + handle + pivot stud + collar + pin, exported as one STEP holding
#    five disjoint solids (the shape a downloaded assembly arrives in).
body = (cq.Workplane("XY").box(60, 40, 12)
        .edges("|Z").fillet(6)
        .faces(">Z").workplane().rarray(44, 26, 2, 2).hole(6.6)
        .faces(">Z").workplane().hole(22))
handle = (cq.Workplane("XY").transformed(offset=(0, 70, 0))
          .box(70, 14, 10).edges("|Z").fillet(5)
          .faces(">Z").workplane().hole(12))
stud = (cq.Workplane("XY").transformed(offset=(0, 110, 0))
        .circle(9).extrude(26).faces(">Z").workplane().circle(5).extrude(10))
collar = (cq.Workplane("XY").transformed(offset=(45, 110, 0))
          .circle(11).extrude(8).faces(">Z").workplane().hole(10))
pin = (cq.Workplane("XY").transformed(offset=(75, 110, 0))
       .circle(3).extrude(28))
asm = body.union(handle).union(stud).union(collar).union(pin)
save(asm, "handle_assembly")
