import { Component,Renderer2 } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { FileUploadModule } from 'primeng/fileupload';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';


interface UploadEvent {
  originalEvent: Event;
  files: File[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule,RouterOutlet,FileUploadModule, FormsModule, ReactiveFormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'DiseaseDetection';
  imageUploader:any = null;
  img:any;
  accuracy:any;
  dName:any;
  errorName:any;
  constructor(public httpclient:HttpClient){}


  onSubmit(){

  }
  onUpload(evt:any){
    let image = evt.srcElement.files[0];
    this.img = image;
    let url = 'http://127.0.0.1:5000/predict';
    let formData = new FormData();
    formData.append('image', this.img);
    this.httpclient.post(url,formData).subscribe((res:any)=>{
      this.accuracy = res['Accuracy'];
      this.dName = res['Type of Disease'];
      this.errorName = res['error'];
      console.log(res);
    },(err:any)=>{
      console.log(err)
    })
    console.log(this.imageUploader);
    console.log(evt);
  }
}
