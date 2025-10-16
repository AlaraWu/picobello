onerror {resume}
quietly WaveActivateNextPane {} 0

delete wave *

set tb_dut picobello_top

set floo_root [exec bender path floo_noc]
# execute the script in $floo_root/hw/tb/wave/wave.tcl
source [file join $floo_root hw tb wave wave.tcl]

set clusters [lsort -dictionary [find instances -recursive -bydu snitch_cluster_wrapper -nodu]]
set l2mem [lsort -dictionary [find instances -recursive -bydu mem_tile -nodu]]

set NumY 4
set NumX [expr [llength $clusters] / $NumY]

configure wave -namecolwidth 150
configure wave -signalnamewidth 1
configure wave -datasetprefix 0

for {set x 0} {$x < $NumX} {incr x} {
  for {set y 0} {$y < $NumY} {incr y} {
    set group_name Mesh_${x}_${y}
    set NumID [expr 4 * $x + $y]
    add wave -noupdate -group $group_name -group Cluster -ports tb_picobello_top/fix/dut/gen_clusters[$NumID]/i_cluster_tile/i_cluster/*
    floo_nw_chimney_wave tb_picobello_top/fix/dut/gen_clusters[$NumID]/i_cluster_tile/i_chimney [list $group_name Chimney]
    add wave -noupdate -group $group_name -group Router -ports tb_picobello_top/fix/dut/gen_clusters[$NumID]/i_cluster_tile/i_router/*
  }
}

set i 0
foreach l2mem $l2mem {
  set group_name L2Mem_$i
  add wave -noupdate -group $group_name -group L2Mem -ports $l2mem/*
  incr i
}