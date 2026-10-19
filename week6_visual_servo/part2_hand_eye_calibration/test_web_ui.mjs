// Run with Node.js; exercises the real page script without hardware or a service.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('./web_ui.html', import.meta.url), 'utf8');
assert.ok(html.indexOf('id="feedback-warning"') < html.indexOf('<header>'));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const jogButtons = [{disabled:false},{disabled:false}];
const element = id => {
  if (!elements.has(id)) elements.set(id, {hidden:true,disabled:false,value:'',textContent:'',innerHTML:'',style:{},children:[],appendChild(child){this.children.push(child);},addEventListener(){}});
  return elements.get(id);
};
let state = {
  ok:true,camera_name:'test',camera_backend:'test',board_found:true,board_ready:true,
  raw:[31,500,500,500,999,1022],capture_blocked_joints:[],can_capture:true,can_solve:false,
  sample_count:0,samples:[],motion_busy:false,motion_active:false,result:null,
};
const context = vm.createContext({
  document:{getElementById:element,createElement:()=>({}),addEventListener(){},querySelectorAll:q=>q==='.jog'?jogButtons:[]},
  window:{addEventListener(){}},location:{protocol:'http:'},setInterval(){},
  fetch:async()=>({json:async()=>state}),navigator:{},console,
});
vm.runInContext(script,context);
await context.refresh();
assert.equal(element('feedback-warning').hidden,true);
assert.equal(element('capture').disabled,false);
state = {...state,raw:[30,500,500,500,1000,1022],can_capture:false,
  capture_blocked_joints:[{id:1,name:'底座',raw:30,interval:'0～30'},
                         {id:5,name:'腕部旋转',raw:1000,interval:'1000～1022'}]};
await context.refresh();
assert.equal(element('feedback-warning').hidden,false);
assert.match(element('feedback-warning-list').innerHTML,/ID 1/);
assert.match(element('feedback-warning-list').innerHTML,/ID 5/);
assert.doesNotMatch(element('feedback-warning-list').innerHTML,/ID 6/);
assert.equal(element('capture').disabled,true);
assert.ok(jogButtons.every(button=>!button.disabled));
state = {...state,raw:[31,500,500,500,999,0],capture_blocked_joints:[],can_capture:true};
await context.refresh();
assert.equal(element('feedback-warning').hidden,true);
assert.equal(element('capture').disabled,false);
state = {...state,raw:null,can_capture:false,read_error:'ID 3 位置反馈读取失败：无回复'};
await context.refresh();
assert.equal(element('feedback-warning').hidden,false);
assert.match(element('feedback-warning-list').innerHTML,/ID 3/);
assert.match(element('feedback-warning-list').innerHTML,/无回复/);
assert.equal(element('capture').disabled,true);
state = {...state,raw:[31,500,500,500,999,0],read_error:'',board_found:false};
await context.refresh();
assert.match(element('feedback-warning-list').innerHTML,/完整的8×6/);
state = {...state,board_found:true};
await context.refresh();
assert.equal(element('feedback-warning').hidden,true);
state = {...state,sample_count:1,samples:[{name:'pose_001',raw:[500,500,500,500,500,0]}]};
await context.refresh();
assert.match(element('samples').innerHTML,/data-delete-name="pose_001"/);
assert.equal(element('delete-all').disabled,false);
state = {...state,sample_count:0,samples:[]};
await context.refresh();
assert.equal(element('delete-all').disabled,true);
assert.equal(element('samples').innerHTML,'');
state = {...state,result:JSON.parse(fs.readFileSync(new URL('./最终标定结果/hand_eye.json', import.meta.url), 'utf8'))};
await context.refresh();
assert.match(element('result').children.at(-1).textContent,/上一轮ID1/);
assert.match(element('result').children.at(-1).textContent,/经验角度映射/);
assert.doesNotMatch(element('result').children.at(-1).textContent,/120 mm输入约束/);
state = {...state,result:{...state.result,distance_measurement:{fit_radius_mm:120}}};
await context.refresh();
assert.match(element('result').children.at(-1).textContent,/120 mm输入约束/);
console.log('PASS: top warning lists blocked IDs, recovers, ignores ID6, and leaves joint control enabled.');
