"""Console interface for reproducible acquisition and model-input preparation."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import requests

from .asos import download_asos, convert_asos
from .demo import make_demo
from .goes import select_goes_c13, download_goes, reproject_goes_netcdf
from .numerical import build_numerical
from .mrms import fetch_mrms, mrms_archive_url
from .collect import fetch_one,fetch_batch
from .dataset import build_supervised
from .timeutil import parse_utc, iso_utc


def bbox_arg(s):
    x=[float(z.strip()) for z in s.split(",")]
    if len(x)!=4:
        raise argparse.ArgumentTypeError("Use south,west,north,east")
    return x


def parser():
    p=argparse.ArgumentParser(prog="weather550",description="CS550 weather data pipeline v0.3")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("demo",help="Fully offline synthetic smoke test")
    a.add_argument("--out",type=Path,default=Path("data/demo"))

    a=sub.add_parser("fetch-asos",help="Download a small real ASOS data sample (Internet required)")
    a.add_argument("--stations",nargs="+",default=["TTN","EWR"])
    a.add_argument("--start",required=True,help="UTC inclusive midnight, e.g. 2026-10-20T00:00:00Z")
    a.add_argument("--end",required=True,help="UTC exclusive midnight, e.g. 2026-10-22T00:00:00Z")
    a.add_argument("--raw",type=Path,default=Path("data/raw/iem_asos.csv"))
    a.add_argument("--out",type=Path,default=Path("data/processed/stations_utc.csv"))

    a=sub.add_parser("convert-asos",help="Convert already downloaded ASOS CSV, no Internet")
    a.add_argument("--raw",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)

    a=sub.add_parser("prepare-numeric",help="Prepare causal numerical tensors, no precipitation targets")
    a.add_argument("--stations",type=Path,required=True)
    a.add_argument("--out",type=Path,default=Path("data/processed/model_numeric.npz"))
    a.add_argument("--start",required=True)
    a.add_argument("--end",required=True)
    a.add_argument("--bbox",type=bbox_arg,default=[39.2,-75.3,41.2,-73.0])
    a.add_argument("--height",type=int,default=32)
    a.add_argument("--width",type=int,default=32)
    a.add_argument("--history",type=int,default=3)
    a.add_argument("--step-minutes",type=int,default=60)
    a.add_argument("--max-age-minutes",type=int,default=90)
    a.add_argument("--latency-minutes",type=int,default=10)
    a.add_argument("--radius-km",type=float,default=110)

    a=sub.add_parser("fetch-goes",help="Fetch ONE real GOES-19 CONUS Band-13 image and regrid to NJ")
    a.add_argument("--at",required=True,help="Scan start near this UTC time")
    a.add_argument("--bucket",default="noaa-goes19")
    a.add_argument("--bbox",type=bbox_arg,default=[39.2,-75.3,41.2,-73.0])
    a.add_argument("--height",type=int,default=32)
    a.add_argument("--width",type=int,default=32)
    a.add_argument("--raw-dir",type=Path,default=Path("data/raw/goes"))
    a.add_argument("--out",type=Path,default=Path("data/processed/goes_C13.npz"))
    a.add_argument("--list-only",action="store_true",help="Find scene metadata without downloading")
    a=sub.add_parser("fetch-mrms",help="Fetch one real historical MRMS 1-hour accumulation grid")
    a.add_argument("--at",required=True,help="UTC exact hour ending accumulation, e.g. 2026-10-20T19:00:00Z")
    a.add_argument("--raw-dir",type=Path,default=Path("data/raw/mrms"))
    a.add_argument("--out-dir",type=Path,default=Path("data/processed/mrms"))
    a.add_argument("--bbox",type=bbox_arg,default=[39.2,-75.3,41.2,-73.0])
    a.add_argument("--height",type=int,default=32)
    a.add_argument("--width",type=int,default=32)
    a.add_argument("--list-only",action="store_true",help="Show source URL without downloading")

    a=sub.add_parser("fetch-sample",help="Fetch 3 GOES scenes and +1/+3h MRMS labels for one issuance")
    a.add_argument("--at",required=True,help="Forecast issue at exact hour, UTC")
    a.add_argument("--bbox",type=bbox_arg,default=[39.2,-75.3,41.2,-73.0])
    a.add_argument("--height",type=int,default=32)
    a.add_argument("--width",type=int,default=32)
    a.add_argument("--list-only",action="store_true")
    a.add_argument("--raw-dir",type=Path,default=Path("data/raw"))
    a.add_argument("--processed-dir",type=Path,default=Path("data/processed"))

    a=sub.add_parser("fetch-batch",help="Bounded incremental retrieval of aligned real samples")
    a.add_argument("--start",required=True)
    a.add_argument("--end",required=True)
    a.add_argument("--step-hours",type=int,default=1)
    a.add_argument("--max-issuances",type=int,default=12)
    a.add_argument("--continue-on-error",action="store_true")
    a.add_argument("--bbox",type=bbox_arg,default=[39.2,-75.3,41.2,-73.0])
    a.add_argument("--height",type=int,default=32)
    a.add_argument("--width",type=int,default=32)
    a.add_argument("--raw-dir",type=Path,default=Path("data/raw"))
    a.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    a.add_argument("--audit",type=Path,default=Path("data/processed/fetch_audit.json"))
    a.add_argument("--list-only",action="store_true")

    a=sub.add_parser("build-supervised",help="Join converted NOAA/ASOS/MRMS files into model-facing tensors")
    a.add_argument("--numeric",type=Path,default=Path("data/processed/model_numeric.npz"))
    a.add_argument("--goes-dir",type=Path,default=Path("data/processed/goes"))
    a.add_argument("--mrms-dir",type=Path,default=Path("data/processed/mrms"))
    a.add_argument("--out",type=Path,default=Path("data/processed/supervised.npz"))
    a.add_argument("--train-before",help="Optional UTC train/val boundary")
    a.add_argument("--val-before",help="Optional UTC val/test boundary")
    a.add_argument("--goes-history",type=int,default=3)
    a.add_argument("--goes-step-minutes",type=int,default=60)
    a.add_argument("--tolerance-minutes",type=int,default=20)
    return p


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try:
        if args.command=="demo":
            result=make_demo(args.out)
        elif args.command=="fetch-asos":
            start,end=parse_utc(args.start),parse_utc(args.end)
            download_asos(args.stations,start,end,args.raw)
            result=convert_asos(args.raw,args.out)
        elif args.command=="convert-asos":
            result=convert_asos(args.raw,args.out)
        elif args.command=="prepare-numeric":
            result=build_numerical(args.stations,args.out,args.start,args.end,
                  bbox=args.bbox,height=args.height,width=args.width,
                  history_steps=args.history,step_minutes=args.step_minutes,
                  max_age_minutes=args.max_age_minutes,radius_km=args.radius_km,
                  latency_minutes=args.latency_minutes)
        elif args.command=="fetch-goes":
            moment=parse_utc(args.at)
            key,scan,available=select_goes_c13(args.bucket,moment,return_availability=True)
            url=f"https://{args.bucket}.s3.amazonaws.com/{key}"
            result={"selected_key":key,"scan_time_utc":iso_utc(scan),"url":url,
                    "available_time_utc":iso_utc(available)}
            if not args.list_only:
                nc=args.raw_dir/Path(key).name
                if not nc.exists():
                    download_goes(args.bucket,key,nc)
                result["processed"]=reproject_goes_netcdf(nc,args.out,bbox=args.bbox,
                        height=args.height,width=args.width,
                        source_url=url,scan_time_utc=iso_utc(scan),available_time_utc=iso_utc(available))
        elif args.command=="fetch-mrms":
            moment=parse_utc(args.at)
            result={"url":mrms_archive_url(moment)} if args.list_only else fetch_mrms(
                moment,args.raw_dir,args.out_dir,bbox=args.bbox,height=args.height,width=args.width)
        elif args.command=="fetch-sample":
            result=fetch_one(parse_utc(args.at),bbox=args.bbox,height=args.height,
                 width=args.width,raw_dir=args.raw_dir,processed_dir=args.processed_dir,
                 list_only=args.list_only)
        elif args.command=="fetch-batch":
            result=fetch_batch(args.start,args.end,step_hours=args.step_hours,
                 max_issuances=args.max_issuances,continue_on_error=args.continue_on_error,
                 bbox=args.bbox,height=args.height,width=args.width,
                 raw_dir=args.raw_dir,processed_dir=args.processed_dir,
                 list_only=args.list_only)
            args.audit.parent.mkdir(parents=True,exist_ok=True)
            args.audit.write_text(json.dumps(result,indent=2),encoding="utf-8")
            result={"audit":str(args.audit),"attempted":result['attempted'],
                    "successes":result['ok'],"failures":result['failures'],
                    "remaining_unattempted":result['remaining_unattempted']}
        elif args.command=="build-supervised":
            result=build_supervised(args.numeric,args.goes_dir,args.mrms_dir,args.out,
               train_before=args.train_before,val_before=args.val_before,
               goes_history=args.goes_history,goes_step_minutes=args.goes_step_minutes,
               tolerance_minutes=args.tolerance_minutes)
        else:
            p.error("Unknown command")
        print(json.dumps(result,indent=2,default=str))
    except (ValueError,RuntimeError,FileNotFoundError,ImportError,requests.RequestException) as ex:
        p.exit(2,f"ERROR: {ex}\n")


if __name__=="__main__":
    main()
