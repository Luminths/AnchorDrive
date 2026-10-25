"""EGO-up map images and lossless first-junction lane branch lookup."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .geometry import corners, point_at, project, wrap
from .road_visualization import drivable_geometry, polygon_parts, preview_body_poses


def _font(size):
    path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path,size)


class LaneCatalog:
    def __init__(self,world,dataset_path):
        self.world=world
        path=Path(dataset_path)/'maps'/'expansion'/(world.map_name+'.json')
        self.drivable_geometry=drivable_geometry(str(path.resolve()))
        raw=json.loads(path.read_text())
        self.ordinary={row['token'] for row in raw['lane']}
        self.connectors={row['token'] for row in raw['lane_connector']}
        self.alias_to_id={};self.id_to_alias={}
        ego=world.states['ego']
        ranked=[]
        for lid,lane in world.lanes.items():
            if lid not in self.ordinary:continue
            _,_,_,distance=project(lane.center.xy,[ego['x'],ego['y']])
            ranked.append((distance,lid))
        for _,lid in sorted(ranked):self.alias(lid)

    def alias(self,lane_id):
        if lane_id not in self.id_to_alias:
            prefix='C' if lane_id in self.connectors else 'L'
            index=1+sum(x.startswith(prefix) for x in self.alias_to_id)
            name=f'{prefix}{index:03d}'
            self.id_to_alias[lane_id]=name;self.alias_to_id[name]=lane_id
        return self.id_to_alias[lane_id]

    def resolve(self,name):
        lane_id=self.alias_to_id.get(name)
        if lane_id is None:raise ValueError('Unknown lane ID '+str(name)+'; copy an L/C ID from the image or a lane query')
        return lane_id

    def ego_xy(self,xy):
        ego=self.world.states['ego'];delta=np.asarray(xy,dtype=float)-[ego['x'],ego['y']]
        c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
        return [round(float(delta@[c,s]),2),round(float(delta@[-s,c]),2)]

    def world_xy(self,forward,left):
        ego=self.world.states['ego'];c,s=math.cos(ego['yaw']),math.sin(ego['yaw'])
        return [ego['x']+forward*c-left*s,ego['y']+forward*s+left*c]

    def heading(self,yaw):
        delta=wrap(yaw-self.world.states['ego']['yaw'])
        directions=['up','upper-left','left','lower-left','down','lower-right','right','upper-right']
        direction=directions[int(math.floor(delta/(math.pi/4)+.5))%8]
        return dict(image_direction=direction,
            angle_convention='degrees relative to EGO heading: 0=up, +90=left, -90=right, +/-180=down',
            unit_vector_ego_forward_left=[round(math.cos(delta),2),round(math.sin(delta),2)],
            relative_heading_deg=round(math.degrees(delta),1))

    def ego_lane_context(self):
        current=self.nearest_lane(self.world.states['ego'])
        if current is None:return dict(current_lane_id=None,map_next_lane_ids=[],map_immediate_exit_lane_ids=[])
        lid=self.resolve(current['lane_id'])
        successors=sorted(self.lane(lid).next_lanes)
        exits=set()
        for next_id in successors:
            if next_id in self.ordinary:exits.add(next_id)
            elif next_id in self.connectors:
                exits.update(x for x in self.lane(next_id).next_lanes if x in self.ordinary)
        return dict(current_lane_id=current['lane_id'],
            current_kind='connector' if lid in self.connectors else 'ordinary_lane',
            map_next_lane_ids=[self.alias(x) for x in successors],
            map_immediate_exit_lane_ids=[self.alias(x) for x in sorted(exits)],
            note='Map next lanes are legal branches, not a committed EGO future route.')

    def nearest_lane(self,actor,ordinary_only=False):
        best=None
        for lid,lane in self.world.lanes.items():
            if ordinary_only and lid not in self.ordinary:continue
            if len(lane.center.xy)<2:continue
            _,_,heading,distance=project(lane.center.xy,[actor['x'],actor['y']])
            error=abs(wrap(heading-actor['yaw']))
            if error>.85 or distance>5:continue
            score=distance+2*error
            if best is None or score<best[0]:best=(score,lid,distance)
        if best is None:return None
        return dict(lane_id=self.alias(best[1]),distance_to_center_m=round(best[2],2))

    def lane(self,lane_id):
        return self.world.vector_map.get_road_lane(lane_id)

    def _points(self,parts,max_points=10):
        pts=np.concatenate([np.asarray(self.lane(lid).center.xy) for lid in parts])
        if len(pts)>max_points:
            indices=np.linspace(0,len(pts)-1,max_points).astype(int)
            pts=pts[indices]
        return [self.ego_xy(p) for p in pts]

    def branches(self,lane_id,include_centerline=True):
        """Return every immediate map-connected route through a junction to an exit lane."""
        start=self.lane(lane_id)
        results=[]

        def walk(current,chain,depth):
            if depth>6 or current in chain:return
            path=chain+[current]
            if current in self.ordinary and current!=lane_id:
                results.append(path);return
            successors=sorted(self.lane(current).next_lanes)
            if not successors:
                results.append(path);return
            for next_id in successors:walk(next_id,path,depth+1)

        for next_id in sorted(start.next_lanes):walk(next_id,[lane_id],1)
        start_points=start.center.xy
        _,start_heading=point_at(start_points,max(0.,start.center.total_distance-2.))
        descriptions=[]
        ego_context=self.ego_lane_context()
        ego_lanes={ego_context['current_lane_id'],*ego_context['map_immediate_exit_lane_ids']}
        for chain in results:
            last=self.lane(chain[-1]);_,end_heading=point_at(last.center.xy,min(3.,last.center.total_distance))
            delta=wrap(end_heading-start_heading)
            turn=('u_turn_or_hairpin' if abs(delta)>2.35 else
                  'left' if delta>.45 else 'right' if delta<-.45 else 'straight_or_merge')
            exit_alias=self.alias(chain[-1])
            reaches_ego=exit_alias in ego_lanes
            descriptions.append(dict(branch_id='B'+str(len(descriptions)+1),
                lane_chain=[self.alias(lid) for lid in chain],
                connector_ids=[self.alias(lid) for lid in chain if lid in self.connectors],
                exit_lane_id=exit_alias,junction_turn_from_lane_end=turn,
                junction_turn_description={'left':'left turn','right':'right turn',
                    'u_turn_or_hairpin':'U-turn or tight hairpin',
                    'straight_or_merge':'straight or merge'}[turn],
                exit_is_ego_current_or_immediate_map_successor=reaches_ego,
                relation_to_ego_corridor=(exit_alias+' is an EGO current/immediate exit lane' if reaches_ego else
                    exit_alias+' is different from EGO '+str(ego_context['current_lane_id'])+
                    ' and its immediate map exits '+','.join(ego_context['map_immediate_exit_lane_ids'])+
                    '; this map branch does not lead into the EGO corridor')))
            if include_centerline:
                descriptions[-1]['centerline_ego_m']=self._points(chain[1:] if len(chain)>1 else chain)
        return descriptions

    def inspect(self,name,actor_alias):
        lane_id=self.resolve(name);lane=self.lane(lane_id)
        points=lane.center.xy
        length=float(lane.center.total_distance)
        occupants=[]
        for aid,actor in self.world.states.items():
            _,_,_,d=project(points,[actor['x'],actor['y']])
            if d<=2.3:
                occupants.append(dict(actor_id=actor_alias(aid),distance_to_center_m=round(d,2),
                    position_ego_m=self.ego_xy([actor['x'],actor['y']])))
        _,entry_yaw=point_at(points,min(1.,length))
        _,exit_yaw=point_at(points,max(0.,length-1.))
        return dict(lane_id=name,kind='connector' if lane_id in self.connectors else 'lane',
            length_m=round(length,1),centerline_ego_m=self._points([lane_id]),
            directed_from_ego_m=self.ego_xy(points[0]),directed_to_ego_m=self.ego_xy(points[-1]),
            entry_heading=self.heading(entry_yaw),exit_heading=self.heading(exit_yaw),
            ego_lane_context=self.ego_lane_context(),
            next_lane_ids=[self.alias(x) for x in sorted(lane.next_lanes)],
            previous_lane_ids=[self.alias(x) for x in sorted(lane.prev_lanes)],
            adjacent_left=[self.alias(x) for x in sorted(lane.adj_lanes_left)],
            adjacent_right=[self.alias(x) for x in sorted(lane.adj_lanes_right)],
            branches=self.branches(lane_id),vehicles_on_lane=occupants,
            note='Coordinates are [EGO forward m, EGO left m]. A branch chain is driven from first ID to last ID. junction_turn_from_lane_end describes only the lane-end turn; the source lane itself may curve or reverse direction. Compare the actor heading with the exit heading for the complete maneuver. These are static map connections, not EGO plans or collision predictions.')

    def visible_ordinary(self,radius):
        ranked=[]
        for lid,lane in self.world.lanes.items():
            if lid not in self.ordinary:continue
            local=np.asarray([self.ego_xy(p) for p in lane.center.xy])
            distance=np.linalg.norm(local,axis=1).min()
            if distance<=radius*1.35:ranked.append((distance,lid))
        return [lid for _,lid in sorted(ranked)]


def render_scene(world,catalog,path,actor_alias,radius_m=55.,highlight_ids=(),focus_lane_id=None,ego_plan=None,anchor_preview=None,ego_history=None,execution_comparison=None,execution_layers=None,camera_ego=None,separate_vehicle_labels=False):
    """Draw the current measured scene with EGO at image center and heading up."""
    size=1200;image=Image.new('RGB',(size,size),'#e7ddd1');draw=ImageDraw.Draw(image)
    scale=size/(2*radius_m);center=size/2
    def screen(xy):
        if camera_ego is None:forward,left=catalog.ego_xy(xy)
        else:
            dx,dy=xy[0]-camera_ego['x'],xy[1]-camera_ego['y'];c,s=math.cos(camera_ego['yaw']),math.sin(camera_ego['yaw'])
            forward,left=c*dx+s*dy,-s*dx+c*dy
        return (center-left*scale,center-forward*scale)
    def line(points,fill,width):
        if len(points)>1:draw.line([screen(p) for p in points],fill=fill,width=width,joint='curve')
    # Full annotated surface, not just lane strips. Clip with a margin so the
    # artificial clipping boundary stays outside the image even after rotation.
    from shapely.geometry import box
    ego=world.states['ego'];extent=radius_m*1.5
    surface=catalog.drivable_geometry.intersection(box(
        ego['x']-extent,ego['y']-extent,ego['x']+extent,ego['y']+extent))
    surface_mask=Image.new('L',image.size,0)
    for polygon in polygon_parts(surface):
        mask=Image.new('L',image.size,0);mask_draw=ImageDraw.Draw(mask)
        mask_draw.polygon([screen(p) for p in polygon.exterior.coords],fill=255)
        for hole in polygon.interiors:
            mask_draw.polygon([screen(p) for p in hole.coords],fill=0)
        surface_mask=ImageChops.lighter(surface_mask,mask)
    image.paste('#f9fbfc',(0,0,size,size),surface_mask)
    # Draw after all fills so islands inside holes remain visible.
    for polygon in polygon_parts(surface):
        for ring in (polygon.exterior, *polygon.interiors):
            line(list(ring.coords),'#955536',3)
    ordinary=catalog.visible_ordinary(radius_m)
    for lid in ordinary:
        lane=catalog.lane(lid)
        for edge in (lane.left_edge,lane.right_edge):
            if edge is not None:line(edge.xy,'#a4b3c1',2)
        line(lane.center.xy,'#cbd5df',2)
    # Connector centerlines are map paths, not painted road markings.
    for lid,lane in world.lanes.items():
        if lid in catalog.connectors:line(lane.center.xy,'#e0e7ee',2)

    ego_context=catalog.ego_lane_context()
    ego_lane_name=ego_context['current_lane_id']
    highlighted_connectors={}
    if ego_lane_name is not None and ego_context['current_kind']=='connector':
        highlighted_connectors[catalog.resolve(ego_lane_name)]='#5685bd'
    for aid in highlight_ids:
        if aid not in world.states:continue
        current=catalog.nearest_lane(world.states[aid])
        if current is not None:
            lane_id=catalog.resolve(current['lane_id'])
            if lane_id in catalog.connectors:highlighted_connectors.setdefault(lane_id,'#e77929')
    connector_labels=[]
    for lane_id,color in highlighted_connectors.items():
        connector_lane=catalog.lane(lane_id)
        line(connector_lane.center.xy,color,4)
        marker,_=point_at(connector_lane.center.xy,connector_lane.center.total_distance*.72)
        mx,my=screen(marker)
        candidates=[(mx+10,my-15,mx+73,my+13),(mx+10,my+20,mx+73,my+48),
                    (mx-75,my-15,mx-12,my+13),(mx-75,my+20,mx-12,my+48)]
        box=next((box for box in candidates if not any(
            not(box[2]<other[0] or box[0]>other[2] or box[3]<other[1] or box[1]>other[3])
            for other in connector_labels)),candidates[-1])
        connector_labels.append(box)
        if box!=candidates[0]:draw.line([(mx,my),(box[0],box[1]+14)],fill=color,width=1)
        draw.rounded_rectangle(box,radius=5,
            fill='#ffffff',outline=color,width=2)
        draw.text((box[0]+6,box[1]+5),catalog.alias(lane_id),font=_font(16),fill=color)

    font=_font(17);small=_font(13);title=_font(22)
    used=list(connector_labels)
    for lid in ordinary:
        lane=catalog.lane(lid);pts=lane.center.xy
        candidates=[pts[i] for i in sorted(set(int(t*(len(pts)-1)) for t in (.25,.5,.75)))]
        for p in candidates:
            x,y=screen(p)
            if not 25<x<size-65 or not 32<y<size-35:continue
            label=catalog.alias(lid);box=(x-3,y-10,x+54,y+13)
            if any(not(box[2]<other[0] or box[0]>other[2] or box[3]<other[1] or box[1]>other[3]) for other in used):continue
            draw.rounded_rectangle(box,radius=4,fill='#ffffff',outline='#7993a8',width=1)
            draw.text((x+3,y-9),label,font=small,fill='#123c58')
            used.append(box)
            break
    for lid in ordinary:
        lane=catalog.lane(lid)
        if len(lane.center.xy)<2:continue
        # Sample across the whole lane so long lanes still show arrows inside the viewport.
        visible_arrows=0
        for distance in np.arange(min(2.,lane.center.total_distance/2),lane.center.total_distance,10.):
            point,heading=point_at(lane.center.xy,float(distance))
            x,y=screen(point);yaw=wrap(heading-world.states['ego']['yaw'])
            forward=(-math.sin(yaw),-math.cos(yaw));length=13
            tip=(x+forward[0]*length,y+forward[1]*length)
            if not (25<x<size-70 and 95<y<size-65):continue
            draw.line([(x-forward[0]*length,y-forward[1]*length),tip],fill='#365e7c',width=3)
            cross=(-forward[1],forward[0])
            draw.polygon([tip,(tip[0]-forward[0]*9+cross[0]*5,tip[1]-forward[1]*9+cross[1]*5),
                (tip[0]-forward[0]*9-cross[0]*5,tip[1]-forward[1]*9-cross[1]*5)],fill='#365e7c')
            if visible_arrows==0:
                label=catalog.alias(lid);tx=x+12;ty=y+8
                bounds=draw.textbbox((tx,ty),label,font=small)
                draw.rectangle((bounds[0]-2,bounds[1]-2,bounds[2]+2,bounds[3]+2),fill='#ffffff')
                draw.text((tx,ty),label,font=small,fill='#123c58')
            visible_arrows+=1

    if focus_lane_id is not None:
        source=catalog.resolve(focus_lane_id)
        line(catalog.lane(source).center.xy,'#075985',6)
        palette=['#dc2626','#8b5cf6','#059669','#d97706','#be185d','#2563eb']
        for index,branch in enumerate(catalog.branches(source)):
            chain=[catalog.resolve(alias) for alias in branch['lane_chain']]
            color=palette[index%len(palette)]
            for lid in chain[1:]:line(catalog.lane(lid).center.xy,color,5)
            exit_lane=catalog.lane(chain[-1])
            point,_=point_at(exit_lane.center.xy,min(exit_lane.center.total_distance,8.))
            bx,by=screen(point)
            if 20<bx<size-70 and 20<by<size-25:
                draw.rounded_rectangle((bx-4,by-12,bx+42,by+14),radius=5,fill='#ffffff',outline=color,width=2)
                draw.text((bx+2,by-9),branch['branch_id'],font=small,fill=color)

    if ego_history is not None:
        past=[p for p in ego_history['points'] if p['t_s']<=world.time+1e-8]
        # Use measured poses, including yaw and dimensions, rather than infer
        # heading from a polyline (which is ambiguous when EGO is stationary).
        overlay=Image.new('RGBA',image.size,(0,0,0,0));body_draw=ImageDraw.Draw(overlay)
        for p in past:
            if 'pose_world' in p:
                body_draw.polygon([screen(c) for c in corners(p['pose_world'])],fill=(21,101,192,22))
        image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
        draw=ImageDraw.Draw(image)
        line([p['world_xy_m'] for p in past],'#1565c0',6)
        for index,p in enumerate(past):
            if index%5 and index!=len(past)-1:continue
            x,y=screen(p['world_xy_m'])
            draw.ellipse((x-3,y-3,x+3,y+3),fill='#1565c0')
            if 'pose_world' not in p:continue
            pose=p['pose_world'];body=[screen(c) for c in corners(pose)]
            draw.line(body+[body[0]],fill='#1565c0',width=2)
            half=pose['length']/2
            tip=(pose['x']+half*math.cos(pose['yaw']),pose['y']+half*math.sin(pose['yaw']))
            line([p['world_xy_m'],tip],'#174197',2)
            if p['t_s']<world.time-1e-8:
                label=f"EGO {p['t_s']-world.time:+.1f}s"
                draw.text((x+9,y-18),label,font=small,fill='#174197',stroke_width=2,stroke_fill='white')

    if ego_plan is not None:
        future=[p for p in ego_plan['points'] if p['t_s']>world.time+1e-8]
        points=[[world.states['ego']['x'],world.states['ego']['y']]]+[p['world_xy_m'] for p in future]
        ego_bodies=ego_plan.get('show_body_preview',False)
        color='#1565c0' if ego_bodies else '#0891b2'
        if ego_bodies:
            overlay=Image.new('RGBA',image.size,(0,0,0,0));body_draw=ImageDraw.Draw(overlay)
            for p in future:
                body_draw.polygon([screen(c) for c in corners(p['pose_world'])],fill=(21,101,192,25))
            image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB');draw=ImageDraw.Draw(image)
        line(points,color,6)
        samples=[p for index,p in enumerate(future,1) if index%5==0 or index==len(future)]
        if not ego_bodies:samples=future[::5]+future[-1:]
        for p in samples:
            x,y=screen(p['world_xy_m'])
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            if ego_bodies:
                pose=p['pose_world'];body=[screen(c) for c in corners(pose)]
                draw.line(body+[body[0]],fill=color,width=2)
                half=pose['length']/2
                tip=(pose['x']+half*math.cos(pose['yaw']),pose['y']+half*math.sin(pose['yaw']))
                line([p['world_xy_m'],tip],color,2)
                draw.text((x+9,y-18),f"EGO +{p['t_s']-world.time:.1f}s",font=small,
                    fill=color,stroke_width=2,stroke_fill='white')
        if future and not ego_bodies:
            x,y=screen(future[-1]['world_xy_m'])
            draw.text((x+8,y-8),f"+{future[-1]['t_s']-world.time:.1f}s",font=small,fill='#08758c')

    if anchor_preview is not None:
        # Translucent .1s body samples show vehicle width, not a safety verdict.
        overlay=Image.new('RGBA',image.size,(0,0,0,0));body_draw=ImageDraw.Draw(overlay)
        preview_poses={aid:preview_body_poses(world.states[aid],plan,world.dt)
                       for aid,plan in anchor_preview.items()}
        for poses in preview_poses.values():
            for _,pose in poses[1:]:
                body_draw.polygon([screen(p) for p in corners(pose)],fill=(147,51,234,35))
        image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
        draw=ImageDraw.Draw(image)
        for aid,plan in anchor_preview.items():
            points=plan['points']
            anchor_times={round(p['t_s'],8) for p in points[1:]}
            for index,(t,pose) in enumerate(preview_poses[aid][1:],1):
                if index!=1 and index%4 and t not in anchor_times:continue
                body=[screen(p) for p in corners(pose)]
                draw.line(body+[body[0]],fill='#a76bd1',width=2)
                x,y=pose['x'],pose['y'];half=pose['length']/2
                tip=(x+half*math.cos(pose['yaw']),y+half*math.sin(pose['yaw']))
                line([(x,y),tip],'#6b21a8',2)
            line([p['position_world_m'] for p in points],'#9333ea',5)
            for p in points[1:]:
                x,y=screen(p['position_world_m'])
                draw.ellipse((x-6,y-6,x+6,y+6),fill='#9333ea',outline='white',width=2)
                label=f"{actor_alias(aid)} +{p['t_s']:.1f}s"
                draw.text((x+9,y-18),label,font=small,fill='#6b21a8',stroke_width=2,stroke_fill='white')

    selected=set(highlight_ids)
    vehicle_labels=[]
    body_bounds=[]
    if separate_vehicle_labels:
        for actor in world.states.values():
            points=[screen(p) for p in corners(actor)]
            body_bounds.append((min(p[0] for p in points),min(p[1] for p in points),
                                max(p[0] for p in points),max(p[1] for p in points)))
    for aid,actor in sorted(world.states.items(),key=lambda pair:pair[0]=='ego'):
        color='#2563eb' if aid=='ego' else '#e77929' if aid in selected else '#728296'
        polygon=[screen(p) for p in corners(actor)]
        draw.polygon(polygon,fill=color,outline='#23384b',width=2)
        local_yaw=wrap(actor['yaw']-(camera_ego or world.states['ego'])['yaw'])
        x,y=screen([actor['x'],actor['y']]);front=(-math.sin(local_yaw),-math.cos(local_yaw))
        tip=(x+front[0]*(actor['length']*scale/2+13),y+front[1]*(actor['length']*scale/2+13))
        draw.line([(x,y),tip],fill='#15283a',width=3)
        cross=(-front[1],front[0])
        draw.polygon([tip,(tip[0]-front[0]*9+cross[0]*6,tip[1]-front[1]*9+cross[1]*6),
            (tip[0]-front[0]*9-cross[0]*6,tip[1]-front[1]*9-cross[1]*6)],fill='#15283a')
        name=actor_alias(aid);tx=x+8;ty=y-32
        if separate_vehicle_labels:
            if -10<=x<=size+10 and -10<=y<=size+10:
                vehicle_labels.append((name,x,y,'#174197' if aid=='ego' else '#96380a' if aid in selected else '#172f45'))
            continue
        bounds=draw.textbbox((tx,ty),name,font=font)
        draw.rounded_rectangle((bounds[0]-4,bounds[1]-2,bounds[2]+4,bounds[3]+3),
            radius=4,fill='#ffffff',outline='#d2dce5')
        draw.text((tx,ty),name,font=font,fill='#174197' if aid=='ego' else '#96380a' if aid in selected else '#172f45')
    if separate_vehicle_labels:
        occupied=[(0,0,size,165),(0,size-78,size,size)]
        def intersection_area(a,b):
            return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        for name,x,y,color in vehicle_labels:
            bounds=draw.textbbox((0,0),name,font=font)
            width,height=bounds[2]-bounds[0]+10,bounds[3]-bounds[1]+6
            candidates=[]
            for dy in (-34,18,-60,44,-86,70,-112,96):
                for dx in (8,-width-8,28,-width-28):
                    left=max(5,min(size-width-5,x+dx));top=max(5,min(size-height-5,y+dy))
                    box=(left,top,left+width,top+height)
                    overlap=sum(intersection_area(box,b) for b in occupied+body_bounds)
                    candidates.append((overlap*1000+abs(left-x)+abs(top-y),box))
            _,box=min(candidates,key=lambda item:item[0])
            occupied.append(box)
            anchor=(max(box[0],min(box[2],x)),max(box[1],min(box[3],y)))
            draw.line([(x,y),anchor],fill=color,width=1)
            draw.rounded_rectangle(box,radius=4,fill='#ffffff',outline='#708597')
            draw.text((box[0]+5-bounds[0],box[1]+3-bounds[1]),name,font=font,fill=color)
    # Draw origins after current vehicles so filled bodies cannot hide them.
    if ego_plan is not None and ego_plan.get('show_body_preview',False):
        origins=[('ego','#1565c0')]
        if anchor_preview is not None:origins.extend((aid,'#9333ea') for aid in anchor_preview)
        for aid,color in origins:
            actor=world.states[aid];x,y=screen([actor['x'],actor['y']])
            draw.ellipse((x-6,y-6,x+6,y+6),fill=color,outline='white',width=2)
            draw.text((x+10,y+10),actor_alias(aid)+' +0.0s',font=small,
                fill=color,stroke_width=2,stroke_fill='white')
    if execution_comparison is not None:
        # Both endpoints use the SAME current EGO camera, so displacements are
        # real measured motion rather than two independently recentered views.
        for aid,poses in execution_comparison['actors'].items():
            color='#075985' if aid=='ego' else '#a3470b'
            for phase in ('before','after'):
                pose=poses[phase];body=[screen(c) for c in corners(pose)]
                if phase=='before':
                    for a,b in zip(body,body[1:]+body[:1]):
                        length=math.hypot(b[0]-a[0],b[1]-a[1]);steps=max(1,math.ceil(length/7))
                        for k in range(0,steps,2):
                            draw.line([(a[0]+(b[0]-a[0])*j/steps,a[1]+(b[1]-a[1])*j/steps)
                                for j in (k,min(k+1,steps))],fill=color,width=3)
                x,y=screen([pose['x'],pose['y']])
                draw.ellipse((x-5,y-5,x+5,y+5),fill='white' if phase=='before' else color,outline=color,width=2)
                if phase=='before':
                    tip=(pose['x']+pose['length']/2*math.cos(pose['yaw']),
                         pose['y']+pose['length']/2*math.sin(pose['yaw']))
                    line([(pose['x'],pose['y']),tip],color,2)
                time_s=execution_comparison['start_s' if phase=='before' else 'end_s']
                label=f"{actor_alias(aid)} {phase.upper()} t={time_s:.1f}s"
                width=draw.textbbox((0,0),label,font=small)[2]
                label_x=max(16,x-width-12) if aid=='ego' else min(size-width-16,x+pose['width']*scale/2+12)
                draw.text((label_x,y+(34 if phase=='after' else -18)),label,
                    font=small,fill=color,stroke_width=2,stroke_fill='white')
    draw.rounded_rectangle((15,12,390,49),radius=6,fill='#ffffff',outline='#b6c5d2')
    draw.text((25,18),'EGO FORWARD  UP  |  LEFT  <-',font=title,fill='#18364d')
    lane_caption='EGO map lane: '+str(ego_lane_name)
    if ego_context['map_next_lane_ids']:
        lane_caption+='  ->  '+','.join(ego_context['map_next_lane_ids'][:3])
    if (ego_context['map_immediate_exit_lane_ids'] and
            ego_context['map_immediate_exit_lane_ids']!=ego_context['map_next_lane_ids']):
        lane_caption+='  ->  '+','.join(ego_context['map_immediate_exit_lane_ids'][:3])
    draw.rounded_rectangle((15,53,475,82),radius=5,fill='#ffffff',outline='#b6c5d2')
    draw.text((24,57),lane_caption,font=font,fill='#174197')
    if ego_plan is not None:
        color='#1565c0' if ego_plan.get('show_body_preview',False) else '#0891b2'
        draw.rounded_rectangle((15,87,710,119),radius=5,fill='#ffffff',outline=color)
        draw.line((26,103,65,103),fill=color,width=5)
        label='EGO forecast (tentative)' if ego_plan.get('show_body_preview',False) or ego_history is not None else 'EGO current plan'
        draw.text((75,91),f"{label} | replans every {ego_plan['replan_interval_s']:g}s",font=font,fill=color)
    elif ego_history is not None:
        draw.rounded_rectangle((15,87,710,119),radius=5,fill='#ffffff',outline='#1565c0')
        draw.line((26,103,65,103),fill='#1565c0',width=5)
        label='EGO executed history (past)' if ego_history['points'] else 'No previous EGO execution yet'
        draw.text((75,91),label+' | replans every 0.1s',font=font,fill='#1565c0')
    draw.rounded_rectangle((15,size-55,240,size-12),radius=6,fill='#ffffff',outline='#b6c5d2')
    draw.line((31,size-34,31+10*scale,size-34),fill='#29475e',width=3)
    draw.text((35+10*scale,size-42),'10 m',font=font,fill='#29475e')
    caption=f"STATE t={world.time:.1f}s"
    if ego_plan is not None and ego_plan.get('show_body_preview',False):caption+=' | +0s = NOW'
    if anchor_preview is not None:caption+=' | PURPLE: PROPOSED WAYPOINTS - NOT EXECUTED'
    draw.rectangle((15,124,1000,158),fill='white')
    draw.text((24,129),caption,font=font,fill='#6b21a8' if anchor_preview is not None else '#18364d')
    draw.rectangle((255,size-77,size-15,size-12),fill='white')
    draw.rectangle((265,size-66,281,size-50),fill='#f9fbfc',outline='#955536',width=2)
    draw.text((290,size-69),'Mapped drivable | brown edge = surface boundary',font=small,fill='#714027')
    draw.rectangle((265,size-43,281,size-27),fill='#e7ddd1')
    legend='Tan = outside mapped surface (not necessarily a physical barrier)'
    draw.text((290,size-46),legend,font=small,fill='#714027')
    if anchor_preview is not None:
        draw.rectangle((15,160,1000,185),fill='white')
        draw.text((24,162),'PURPLE BODIES: sampled proposed footprint + heading; check the full width',font=small,fill='#6b21a8')
    if ego_plan is not None and ego_plan.get('show_body_preview',False):
        y=187 if anchor_preview is not None else 160
        draw.rectangle((15,y,1000,y+25),fill='white')
        draw.text((24,y+2),'BLUE BODIES: tentative EGO future | both paths start NOW; future may change',font=small,fill='#1565c0')
    elif ego_history is not None and ego_plan is not None:
        draw.rectangle((15,187,1000,212),fill='white')
        draw.text((24,189),'BLUE: past EGO execution | CYAN: tentative forecast; closed-loop future may change',font=small,fill='#1565c0')
    elif ego_history is not None:
        y=187 if anchor_preview is not None else 160
        draw.rectangle((15,y,1000,y+25),fill='white')
        draw.text((24,y+2),'BLUE BODIES: measured past footprint + heading; negative times = history',font=small,fill='#1565c0')
    if execution_comparison is not None:
        draw.rectangle((15,214,1000,239),fill='white')
        draw.text((24,216),'MEASURED LAST INTERVAL: BEFORE dashed / AFTER solid (NOW); t = simulation time',font=small,fill='#18364d')
    if execution_layers is not None:
        from .trial_waypoint_execution import draw_execution_layers
        draw_execution_layers(draw,screen,execution_layers,actor_alias,small,line,size)
    image.save(path)
    return [catalog.alias(lid) for lid in ordinary]
